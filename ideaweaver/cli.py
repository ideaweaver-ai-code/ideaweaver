#!/usr/bin/env python3
"""
IdeaWeaver CLI - Enhanced with Langfuse monitoring
"""

import click
#from .langfuse_integration import IdeaWeaverLangfuse
#from .langfuse_cli import langfuse

@click.group()
def cli():
    """IdeaWeaver Model Training CLI - A comprehensive tool for AI model training, evaluation, and deployment.
    
    Features include LoRA/QLoRA fine-tuning, RAG systems, MCP integration, and enterprise-grade model management.
    For detailed documentation and examples, visit: https://github.com/ideaweaver-ai-code/ideaweaver
    """
    pass

# Add Langfuse commands unconditionally
# cli.add_command(langfuse)

# Initialize Langfuse silently
try:
    from langfuse import Langfuse
    langfuse_client = Langfuse(
        public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
        secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
        host=os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')
    )
except Exception:
    langfuse_client = None

# Suppress Langfuse warning
import warnings
warnings.filterwarnings('ignore', message='.*Langfuse client is disabled.*')

import os
import sys
import time
import json
import asyncio
import re
import warnings
import io
import contextlib
import torch

# Simple and effective warning suppression
warnings.filterwarnings("ignore", message=".*'NoneType' object has no attribute 'cadam32bit_grad_fp32'.*")
warnings.filterwarnings("ignore", message=".*cadam32bit_grad_fp32.*")
warnings.filterwarnings("ignore", message=".*comet_ml.*")
warnings.filterwarnings("ignore", message=".*Comet.*API.*Key.*")
warnings.filterwarnings("ignore", message=".*bitsandbytes.*")
warnings.filterwarnings("ignore", message=".*pydantic.*ForwardRef.*")
warnings.filterwarnings("ignore", message=".*urllib3.*socks.*")

# Environment variables to reduce noise
os.environ.setdefault("COMET_DISABLE_AUTO_LOGGING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Simple stderr filter only for the specific problematic message
class SimpleWarningFilter:
    def __init__(self, original_stream):
        self.original_stream = original_stream
        self.problematic_patterns = [
            "'NoneType' object has no attribute 'cadam32bit_grad_fp32'",
            "comet_ml is installed but the Comet API Key is not configured",
            "DeprecationWarning: Failing to pass a value to the 'type_params' parameter",
            "SentryHubDeprecationWarning:",
            "sentry_sdk.Hub",
            "PEP 695 type parameter",
            "typing.ForwardRef._evaluate",
            "migration guide for details on how to migrate"
        ]
    
    def write(self, text):
        # Only filter out the specific problematic lines
        for pattern in self.problematic_patterns:
            if pattern in text:
                return  # Don't write this line
        self.original_stream.write(text)
        self.original_stream.flush()
    
    def flush(self):
        self.original_stream.flush()
    
    def __getattr__(self, name):
        return getattr(self.original_stream, name)

# Only filter stderr for the specific messages
sys.stderr = SimpleWarningFilter(sys.stderr)
# Also filter stdout to catch warnings printed there
sys.stdout = SimpleWarningFilter(sys.stdout)

@contextlib.contextmanager
def suppress_stderr():
    """Temporarily suppress stderr"""
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stderr = old_stderr

# Apply immediate warning suppression
with suppress_stderr():
    # Targeted warning suppression - only suppress specific problematic warnings
    # Keep important warnings like security, API changes, etc.
    
    # Suppress only the specific annoying warnings that don't affect functionality
    warnings.filterwarnings("ignore", message=".*bitsandbytes.*")
    warnings.filterwarnings("ignore", message=".*GPU support.*")
    warnings.filterwarnings("ignore", message=".*cadam32bit_grad_fp32.*")
    warnings.filterwarnings("ignore", message=".*'NoneType' object has no attribute 'cadam32bit_grad_fp32'.*")
    warnings.filterwarnings("ignore", message=".*comet_ml.*")
    warnings.filterwarnings("ignore", message=".*Comet.*")
    warnings.filterwarnings("ignore", message=".*COMET_API_KEY.*")
    warnings.filterwarnings("ignore", message=".*NoneType.*cadam32bit_grad_fp32.*")
    
    # Suppress specific library warnings that are noisy but not important
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*transformers.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module=".*torch.*")
    warnings.filterwarnings("ignore", category=UserWarning, module=".*bitsandbytes.*")
    warnings.filterwarnings("ignore", category=UserWarning, module=".*comet_ml.*")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module=".*torch.*")
    
    # Environment variables for specific libraries
    os.environ.setdefault("COMET_DISABLE_AUTO_LOGGING", "1")
    os.environ.setdefault("COMET_LOGGING_DISABLE", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    
    # Only suppress logs for specific noisy libraries
    import logging
    logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
    logging.getLogger("comet_ml").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)

# Allow users to enable warnings for debugging by setting environment variable
if os.environ.get("IDEAWEAVER_SHOW_WARNINGS", "").lower() in ("1", "true", "yes"):
    # Reset warnings to default behavior
    warnings.resetwarnings()
    click.echo("⚠️  Warnings enabled for debugging (IDEAWEAVER_SHOW_WARNINGS=1)")
else:
    # Apply targeted warning suppression - preserve important warnings
    warnings.simplefilter("default")  # Use default behavior
    
    # Add more specific filters for problematic patterns
    warnings.filterwarnings("ignore", message=".*urllib3.*socks.*")
    warnings.filterwarnings("ignore", message=".*pysocks.*")
    warnings.filterwarnings("ignore", category=ImportWarning)

from .trainer import ModelTrainer
from .config import load_config
from .evaluator import LLMEvaluator, create_evaluation_report
from .rag import RAGManager
from .rag_evaluator import RAGEvaluator, RAGEvaluationConfig
from .agentic_rag import AgenticRAGManager, AgenticRAGConfig
from .fine_tuner import SupervisedFineTuner, FineTuningConfig, create_fine_tuning_config
from .mcp import MCPManager, MCPServerConfig, get_mcp_manager, execute_mcp_tool, format_mcp_result
from pathlib import Path

def _suppress_all_warnings():
    """Additional runtime warning suppression"""
    import warnings
    import os
    import logging
    
    # Suppress all warnings
    warnings.simplefilter("ignore")
    
    # Set environment variables
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    
    # Suppress specific loggers
    for logger_name in ["bitsandbytes", "comet_ml", "transformers", "torch", "huggingface_hub"]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

def _import_fine_tuner():
    """Lazy import fine-tuning dependencies only when needed"""
    try:
        from .fine_tuner import SupervisedFineTuner, FineTuningConfig, create_fine_tuning_config
        return SupervisedFineTuner, FineTuningConfig, create_fine_tuning_config
    except ImportError as e:
        click.echo(f"❌ Fine-tuning dependencies not available: {e}", err=True)
        click.echo("Install fine-tuning dependencies: pip install transformers peft torch", err=True)
        sys.exit(1)

@cli.command()
@click.option('--config', '-c', help='Path to YAML config file')
@click.option('--model', '-m', help='Hugging Face model name (e.g., bert-base-uncased, distilbert-base-uncased)')
@click.option('--task', '-t', 
              type=click.Choice(['text_classification', 'text_regression', 'token_classification', 'question_answering']),
              help='Task type')
@click.option('--dataset', '-d', help='Path to dataset file')
@click.option('--project-name', '-p', help='Project name for output')
@click.option('--epochs', type=int, help='Number of training epochs')
@click.option('--batch-size', type=int, help='Training batch size')
@click.option('--learning-rate', type=float, help='Learning rate')
@click.option('--max-seq-length', type=int, help='Maximum sequence length for tokenization')
@click.option('--push-to-hub', is_flag=True, help='Push trained model to Hugging Face Hub')
@click.option('--hub-model-id', help='Hugging Face Hub model ID (username/model-name)')
@click.option('--hf-token', help='Hugging Face token for authentication')
@click.option('--push-to-bedrock', is_flag=True, help='Push trained model to AWS Bedrock')
@click.option('--bedrock-model-name', help='AWS Bedrock model name')
@click.option('--bedrock-s3-bucket', help='S3 bucket for Bedrock model storage')
@click.option('--bedrock-role-arn', help='IAM role ARN for Bedrock import')
@click.option('--bedrock-region', default='us-east-1', help='AWS region for Bedrock (us-east-1, us-west-2, eu-central-1)')
@click.option('--bedrock-s3-prefix', help='S3 prefix/folder for model files')
@click.option('--bedrock-job-name', help='Custom Bedrock import job name')
@click.option('--bedrock-test-inference', is_flag=True, default=True, help='Test model inference after import')
@click.option('--track-experiments', is_flag=True, help='Enable experiment tracking')
@click.option('--comet-api-key', help='Comet ML API key for experiment tracking')
@click.option('--comet-project', help='Comet ML project name')
@click.option('--mlflow-uri', help='MLflow tracking server URI (for DagsHub or custom MLflow)')
@click.option('--mlflow-experiment', help='MLflow experiment name')
@click.option('--dagshub-token', help='DagsHub token for authentication')
@click.option('--dagshub-repo-owner', help='DagsHub repository owner')
@click.option('--dagshub-repo-name', help='DagsHub repository name')
@click.option('--register-model', is_flag=True, help='Automatically register model in MLflow Model Registry (for DagsHub/MLflow) or Comet Model Registry')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--monitor/--no-monitor', default=True, help='Enable Langfuse monitoring')
def train(config, model, task, dataset, project_name, epochs, batch_size, learning_rate, 
          max_seq_length, push_to_hub, hub_model_id, hf_token, 
          push_to_bedrock, bedrock_model_name, bedrock_s3_bucket, bedrock_role_arn,
          bedrock_region, bedrock_s3_prefix, bedrock_job_name, bedrock_test_inference,
          track_experiments, comet_api_key, comet_project, mlflow_uri, mlflow_experiment, 
          dagshub_token, dagshub_repo_owner, dagshub_repo_name, register_model, verbose, monitor):
    """Train a model with AutoTrain Advanced.

    This command provides a powerful interface for training models using AutoTrain Advanced,
    supporting various tasks like text classification, regression, and question answering.
    It includes features for experiment tracking, model deployment, and performance monitoring.
    """
    # If config file provided, load it as base
    if config:
        if not os.path.exists(config):
            click.echo(f"❌ Config file not found: {config}", err=True)
            sys.exit(1)
        config_data = load_config(config)
    else:
        # Create minimal config with defaults
        config_data = {
            'backend': 'local',
            'params': {
                'epochs': 3,
                'batch_size': 8,
                'learning_rate': 2e-5,
                'max_seq_length': 128,
            },
            'data': {'train_split': 'train'},
            'hub': {'push_to_hub': False}
        }
    
    # Override config with command line options
    if model:
        config_data['base_model'] = model
        click.echo(f"🤗 Using model: {model}")
    elif 'base_model' not in config_data:
        config_data['base_model'] = 'google/bert_uncased_L-2_H-128_A-2'  # default
    
    if task:
        config_data['task'] = task
        click.echo(f"🎯 Task: {task}")
    elif 'task' not in config_data:
        config_data['task'] = 'text_classification'  # default
    
    if dataset:
        config_data['dataset'] = dataset
    elif 'dataset' not in config_data:
        click.echo("❌ Dataset is required. Use --dataset or specify in config file.", err=True)
        sys.exit(1)
    
    if project_name:
        config_data['project_name'] = project_name
    elif 'project_name' not in config_data:
        config_data['project_name'] = 'my-model'
    
    # Training parameters
    if epochs:
        config_data['params']['epochs'] = epochs
    if batch_size:
        config_data['params']['batch_size'] = batch_size
    if learning_rate:
        config_data['params']['learning_rate'] = learning_rate
    if max_seq_length:
        config_data['params']['max_seq_length'] = max_seq_length
    
    # Hugging Face Hub settings
    if push_to_hub:
        config_data['hub']['push_to_hub'] = True
        if hub_model_id:
            config_data['hub']['hub_model_id'] = hub_model_id
            # Extract username from hub_model_id (format: username/model-name)
            if '/' in hub_model_id:
                username = hub_model_id.split('/')[0]
                config_data['hub']['username'] = username
            else:
                click.echo("❌ --hub-model-id must be in format: username/model-name", err=True)
                sys.exit(1)
        else:
            click.echo("❌ --hub-model-id required when using --push-to-hub", err=True)
            sys.exit(1)
        
        if hf_token:
            config_data['hub']['token'] = hf_token
            os.environ['HF_TOKEN'] = hf_token
        else:
            # Check if token is in environment
            env_token = os.environ.get('HF_TOKEN')
            if env_token:
                config_data['hub']['token'] = env_token
            else:
                # Set to None - IdeaWeaver will try to use saved credentials
                config_data['hub']['token'] = None
        
        click.echo(f"🚀 Will push to Hub: {hub_model_id}")
    
    # AWS Bedrock settings
    if push_to_bedrock:
        required_bedrock_params = ['bedrock_model_name', 'bedrock_s3_bucket', 'bedrock_role_arn']
        missing_bedrock_params = [p for p in required_bedrock_params if not locals()[p]]
        
        if missing_bedrock_params:
            click.echo("❌ Missing required Bedrock parameters:", err=True)
            for param in missing_bedrock_params:
                click.echo(f"   --{param.replace('_', '-')}", err=True)
            sys.exit(1)
        
        config_data['bedrock'] = {
            'model_name': bedrock_model_name,
            's3_bucket': bedrock_s3_bucket,
            'role_arn': bedrock_role_arn,
            'region': bedrock_region,
            's3_prefix': bedrock_s3_prefix,
            'job_name': bedrock_job_name,
            'test_inference': bedrock_test_inference
        }
        
        click.echo(f"☁️  Will deploy to AWS Bedrock:")
        click.echo(f"   Model name: {bedrock_model_name}")
        click.echo(f"   S3 bucket: {bedrock_s3_bucket}")
        click.echo(f"   Region: {bedrock_region}")
        
        # Check for AWS credentials
        aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        aws_profile = os.environ.get('AWS_PROFILE')
        
        if not any([aws_access_key, aws_profile]):
            click.echo("⚠️  Warning: AWS credentials not found in environment", err=True)
            click.echo("   Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, or", err=True)
            click.echo("   Set AWS_PROFILE, or run 'aws configure'", err=True)
    
    # Experiment tracking settings
    if track_experiments or mlflow_uri or dagshub_token:
        config_data['tracking'] = {
            'enabled': True,
            'comet': {
                'api_key': comet_api_key or os.environ.get('COMET_API_KEY'),
                'project': comet_project or config_data['project_name']
            },
            'mlflow': {
                'uri': mlflow_uri or os.environ.get('MLFLOW_TRACKING_URI'),
                'experiment': mlflow_experiment or config_data['project_name']
            },
            'register_model': register_model
        }
        
        # Add DagsHub configuration if provided
        if dagshub_repo_owner and dagshub_repo_name:
            config_data['tracking']['dagshub'] = {
                'enabled': True,
                'repo_owner': dagshub_repo_owner,
                'repo_name': dagshub_repo_name,
                'token': dagshub_token or os.environ.get('DAGSHUB_TOKEN')
            }
            click.echo("🔗 DagsHub tracking enabled")
            if dagshub_token:
                os.environ['DAGSHUB_TOKEN'] = dagshub_token
        
        click.echo("📊 Experiment tracking enabled")
        if register_model:
            click.echo("🏷️  Model registration enabled (MLflow/DagsHub/Comet)")
    else:
        config_data['tracking'] = {'enabled': False}
    
    # Validate dataset exists
    if not os.path.exists(config_data['dataset']):
        click.echo(f"❌ Dataset file not found: {config_data['dataset']}", err=True)
        sys.exit(1)
    
    try:
        if verbose:
            click.echo("📋 Final configuration:")
            for key, value in config_data.items():
                click.echo(f"   {key}: {value}")
        
        # Initialize trainer
        trainer = ModelTrainer(config_data, verbose=verbose)
        
        # Start training
        click.echo("🚀 Starting model training...")
        result = trainer.train()
        
        if result:
            click.echo("✅ Training completed successfully!")
            click.echo(f"📁 Model saved to: {result}")
            # Register model in MLflow if requested and mlflow_uri is set
            if register_model and (mlflow_uri or (config_data.get('tracking', {}).get('mlflow', {}).get('uri'))):
                click.echo("🏷️  Registering model in MLflow Model Registry...")
                try:
                    import mlflow
                    # Ensure MLflow tracking URI is set
                    if mlflow_uri:
                        mlflow.set_tracking_uri(mlflow_uri)
                    # Get the active run or create a new one
                    active_run = mlflow.active_run()
                    if not active_run:
                        # Create a new run if none exists
                        mlflow.set_experiment(config_data.get('project_name', 'default'))
                        active_run = mlflow.start_run(run_name=f"ideaweaver-{config_data.get('project_name', 'default')}")
                    
                    if active_run:
                        trainer.register_model_in_mlflow(active_run)
                        click.echo("✅ Model registered in MLflow Model Registry!")
                    else:
                        click.echo("❌ Failed to create MLflow run for model registration.", err=True)
                except Exception as e:
                    click.echo(f"❌ Failed to register model in MLflow: {e}", err=True)
            if config_data['hub']['push_to_hub']:
                click.echo("📤 Pushing to Hugging Face Hub...")
                # The trainer should handle the push
        else:
            click.echo("❌ Training failed!")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@cli.command()
@click.argument('model_name')
@click.option('--save-path', default='./downloaded_model', help='Path to save downloaded model')
def download(model_name, save_path):
    """Download a model from Hugging Face Hub"""
    try:
        from transformers import AutoTokenizer, AutoModel
        
        click.echo(f"📥 Downloading model: {model_name}")
        
        # Download tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Save locally
        os.makedirs(save_path, exist_ok=True)
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        
        click.echo(f"✅ Model downloaded to: {save_path}")
        click.echo(f"📝 You can now use: --model {save_path}")
        
    except Exception as e:
        click.echo(f"❌ Download failed: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--config', '-c', required=True, help='Path to YAML config file')
def validate(config):
    """Validate a configuration file"""
    
    if not os.path.exists(config):
        click.echo(f"❌ Config file not found: {config}", err=True)
        sys.exit(1)
    
    try:
        config_data = load_config(config)
        click.echo("✅ Configuration is valid!")
        
        # Check dataset exists
        if 'dataset' in config_data and not os.path.exists(config_data['dataset']):
            click.echo(f"⚠️  Warning: Dataset file not found: {config_data['dataset']}")
        
        # Check model availability
        if 'base_model' in config_data:
            try:
                from transformers import AutoTokenizer
                AutoTokenizer.from_pretrained(config_data['base_model'])
                click.echo(f"✅ Model {config_data['base_model']} is accessible")
            except:
                click.echo(f"⚠️  Warning: Model {config_data['base_model']} may not be accessible")
        
    except Exception as e:
        click.echo(f"❌ Invalid configuration: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('model_path')
@click.option('--tasks', '-t', help='Comma-separated list of tasks to evaluate on')
@click.option('--benchmark-suite', '-s', 
              type=click.Choice(['standard', 'reasoning', 'knowledge', 'comprehensive', 'custom']),
              default='standard',
              help='Predefined benchmark suite to use')
@click.option('--wandb-project', '-wp', default='llm-evaluation', help='Weights & Biases project name')
@click.option('--wandb-entity', '-we', help='Weights & Biases entity name')
@click.option('--device', '-d', default='auto', help='Device to run evaluation on (auto, cuda, cpu)')
@click.option('--batch-size', '-b', type=int, default=8, help='Batch size for evaluation')
@click.option('--num-fewshot', '-f', type=int, help='Number of few-shot examples')
@click.option('--limit', '-l', type=int, help='Limit number of samples for testing')
@click.option('--output-path', '-o', help='Path to save evaluation results')
@click.option('--generate-report', is_flag=True, help='Generate markdown evaluation report')
@click.option('--report-to', type=click.Choice(['wandb', 'tensorboard', 'both', 'none']), 
              default='none', help='Experiment tracking platform')
@click.option('--tensorboard-project', default='ideaweaver-evaluation', 
              help='TensorBoard project name')
@click.option('--tensorboard-experiment', help='Custom TensorBoard experiment name')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def evaluate(model_path, tasks, benchmark_suite, wandb_project, wandb_entity, device, 
             batch_size, num_fewshot, limit, output_path, generate_report, report_to,
             tensorboard_project, tensorboard_experiment, verbose):
    """Evaluate a language model on various benchmarks."""
    try:
        click.echo(f"🚀 Starting LLM evaluation for model: {model_path}")
        if report_to != 'none':
            click.echo(f"📊 Tracking with: {report_to}")
        
        # Initialize TensorBoard tracker if requested
        tensorboard_tracker = None
        if report_to in ['tensorboard', 'both']:
            try:
                from .tensorboard_integration import create_tensorboard_tracker, check_tensorboard_availability
                
                if not check_tensorboard_availability():
                    click.echo("⚠️  TensorBoard not available. Install with: pip install tensorboard")
                    if report_to == 'tensorboard':
                        click.echo("❌ Cannot proceed without TensorBoard. Exiting...")
                        sys.exit(1)
                else:
                    # Create evaluation config for logging
                    eval_config = {
                        "model_path": model_path,
                        "tasks": tasks,
                        "benchmark_suite": benchmark_suite,
                        "device": device,
                        "batch_size": batch_size,
                        "num_fewshot": num_fewshot,
                        "limit": limit
                    }
                    
                    tensorboard_tracker = create_tensorboard_tracker(
                        model_name=model_path,
                        experiment_name=tensorboard_experiment,
                        config=eval_config
                    )
                    
                    click.echo(f"✅ TensorBoard tracking initialized")
                    click.echo(f"   📁 Log directory: {tensorboard_tracker.get_log_dir()}")
                    
            except ImportError:
                click.echo("⚠️  TensorBoard integration module not found")
                if report_to == 'tensorboard':
                    sys.exit(1)
        
        # Initialize evaluator
        use_wandb = report_to in ['wandb', 'both']
        evaluator = LLMEvaluator(
            model_path=model_path,
            wandb_project=wandb_project if use_wandb else None,
            wandb_entity=wandb_entity if use_wandb else None,
            device=device,
            batch_size=batch_size,
            verbose=verbose
        )
        
        # Determine tasks to run
        if tasks:
            # Custom task list
            task_list = [task.strip() for task in tasks.split(',')]
            results = evaluator.evaluate_on_benchmarks(
                tasks=task_list,
                num_fewshot=num_fewshot,
                limit=limit,
                output_path=output_path
            )
        else:
            # Use benchmark suite
            results = evaluator.run_comprehensive_evaluation(
                benchmark_suite=benchmark_suite,
                num_fewshot=num_fewshot,
                limit=limit,
                output_path=output_path
            )
        
        # Log to TensorBoard if enabled
        if tensorboard_tracker:
            tensorboard_tracker.log_evaluation_results(results)
            
            # Log system information
            import platform
            import torch
            system_info = {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "device_name": str(device),
                "torch_version": torch.__version__ if hasattr(torch, '__version__') else "unknown"
            }
            tensorboard_tracker.log_system_info(system_info)
        
        click.echo("✅ Evaluation completed successfully!")
        
        # Generate report if requested
        if generate_report:
            if tensorboard_tracker:
                from .tensorboard_integration import TensorBoardEvaluationReporter
                report_path = TensorBoardEvaluationReporter.create_evaluation_report(
                    results, tensorboard_tracker, model_path
                )
                click.echo(f"📄 Enhanced report saved to: {report_path}")
            else:
                report_path = f"evaluation_report_{model_path.replace('/', '_')}.md"
                create_evaluation_report(results, report_path)
                click.echo(f"📄 Report saved to: {report_path}")
        
        # Print summary
        if "results" in results:
            click.echo("\n📊 Evaluation Summary:")
            for task_name, task_results in results["results"].items():
                for metric_name, metric_value in task_results.items():
                    if isinstance(metric_value, (int, float)):
                        click.echo(f"   {task_name} - {metric_name}: {metric_value:.4f}")
        
        # Finalize tracking
        if tensorboard_tracker:
            # Prepare final metrics
            final_metrics = {}
            if "results" in results:
                for task_name, task_results in results["results"].items():
                    for metric_name, metric_value in task_results.items():
                        if isinstance(metric_value, (int, float)):
                            final_metrics[f"{task_name}_{metric_name}"] = metric_value
            
            tensorboard_tracker.finalize(final_metrics)
            
            # Show TensorBoard URL
            click.echo(f"\n🌐 View results in TensorBoard:")
            click.echo(f"   tensorboard --logdir=tensorboard_logs")
            click.echo(f"   Then open: http://localhost:6006")
        
        # Finish wandb run if used
        if use_wandb:
            evaluator.finish()
            
    except Exception as e:
        click.echo(f"❌ Error during evaluation: {str(e)}")
        if verbose:
            import traceback
            click.echo(traceback.format_exc())
        sys.exit(1)

@cli.command('list-tasks')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def list_tasks(verbose):
    """List all available evaluation tasks from lm-evaluation-harness"""
    
    try:
        # Create a dummy evaluator to get available tasks
        evaluator = LLMEvaluator(
            model_path="gpt2",  # Use a small default model
            verbose=False
        )
        
        tasks = evaluator.get_available_tasks()
        
        if tasks:
            click.echo("📋 Available evaluation tasks:")
            click.echo("=" * 50)
            
            # Group tasks by category if possible
            standard_tasks = ["hellaswag", "arc_easy", "arc_challenge", "winogrande", "piqa"]
            reasoning_tasks = ["mathqa", "gsm8k"]
            knowledge_tasks = ["mmlu", "truthfulqa_mc1", "truthfulqa_mc2"]
            
            def print_task_group(title, task_list):
                group_tasks = [t for t in tasks if t in task_list]
                if group_tasks:
                    click.echo(f"\n{title}:")
                    for task in group_tasks:
                        click.echo(f"  • {task}")
                    return group_tasks
                return []
            
            # Print categorized tasks
            used_tasks = []
            used_tasks.extend(print_task_group("📚 Standard Benchmarks", standard_tasks))
            used_tasks.extend(print_task_group("🧠 Reasoning Tasks", reasoning_tasks))
            used_tasks.extend(print_task_group("🎓 Knowledge Tasks", knowledge_tasks))
            
            # Print remaining tasks
            other_tasks = [t for t in tasks if t not in used_tasks]
            if other_tasks:
                click.echo(f"\n🔧 Other Tasks:")
                for task in other_tasks:
                    click.echo(f"  • {task}")
            
            click.echo(f"\n📊 Total tasks available: {len(tasks)}")
            
            if verbose:
                click.echo("\n💡 Usage examples:")
                click.echo("  # Evaluate on standard suite:")
                click.echo("  ideaweaver-model-train evaluate model_name --benchmark-suite standard")
                click.echo("\n  # Evaluate on specific tasks:")
                click.echo("  ideaweaver-model-train evaluate model_name --tasks hellaswag,arc_easy")
                click.echo("\n  # Compare models:")
                click.echo("  ideaweaver-model-train compare model1 model2 --tasks hellaswag,piqa")
        else:
            click.echo("❌ No tasks found. Make sure lm-evaluation-harness is installed.")
        
        # Finish evaluator
        evaluator.finish()
        
    except Exception as e:
        click.echo(f"❌ Failed to list tasks: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@cli.group()
def rag():
    """RAG (Retrieval-Augmented Generation) commands"""
    pass

@rag.command('create-kb')
@click.option('--name', '-n', required=True, help='Knowledge base name')
@click.option('--description', '-d', default='', help='Knowledge base description')
@click.option('--embedding-model', '-e', default='sentence-transformers/all-MiniLM-L6-v2', 
              help='Embedding model to use')
@click.option('--chunk-size', '-c', type=int, default=512, help='Chunk size for text splitting')
@click.option('--chunk-overlap', '-o', type=int, default=50, help='Overlap between chunks')
@click.option('--chunking-strategy', type=click.Choice(['recursive', 'token', 'semantic']), 
              default='recursive', help='Text chunking strategy')
@click.option('--vector-store', type=click.Choice(['chroma', 'qdrant_local', 'qdrant_cloud']),
              default='chroma', help='Vector store backend')
@click.option('--qdrant-url', help='Qdrant Cloud URL (e.g., https://xyz.qdrant.io)')
@click.option('--qdrant-api-key', help='Qdrant Cloud API key')
@click.option('--qdrant-collection', help='Qdrant collection name (auto-generated if not specified)')
@click.option('--qdrant-grpc', is_flag=True, help='Use gRPC for Qdrant (better performance)')
@click.option('--enable-hybrid-search', is_flag=True, help='Enable hybrid search (semantic + keyword)')
@click.option('--hybrid-alpha', type=float, default=0.5, 
              help='Hybrid search weight (0=keyword only, 1=semantic only)')
@click.option('--enable-reranking', is_flag=True, help='Enable cross-encoder reranking')
@click.option('--reranker-model', default='cross-encoder/ms-marco-MiniLM-L-12-v2',
              help='Cross-encoder model for reranking')
@click.option('--reranker-top-k', type=int, default=20, 
              help='Number of docs to retrieve before reranking')
@click.option('--semantic-threshold', type=float, default=0.8,
              help='Similarity threshold for semantic chunking')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def create_kb(name, description, embedding_model, chunk_size, chunk_overlap, 
              chunking_strategy, vector_store, qdrant_url, qdrant_api_key, 
              qdrant_collection, qdrant_grpc, enable_hybrid_search, hybrid_alpha, 
              enable_reranking, reranker_model, reranker_top_k, semantic_threshold, verbose):
    """Create a new knowledge base with advanced features and cloud support"""
    
    try:
        # Validate cloud configuration
        if vector_store == 'qdrant_cloud':
            if not qdrant_url or not qdrant_api_key:
                click.echo("❌ Qdrant Cloud requires --qdrant-url and --qdrant-api-key", err=True)
                click.echo("💡 Get your credentials from: https://cloud.qdrant.io/", err=True)
                sys.exit(1)
            
            if not qdrant_url.startswith('https://'):
                click.echo("❌ Qdrant Cloud URL must start with https://", err=True)
                sys.exit(1)
        
        rag_manager = RAGManager(verbose=verbose)
        
        # Create basic knowledge base
        config = rag_manager.create_knowledge_base(
            name=name,
            description=description,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Update with advanced features and cloud config
        config.chunking_strategy = chunking_strategy
        config.vector_store = vector_store
        config.use_hybrid_search = enable_hybrid_search
        config.hybrid_alpha = hybrid_alpha
        config.use_reranking = enable_reranking
        config.reranker_model = reranker_model
        config.reranker_top_k = reranker_top_k
        config.semantic_similarity_threshold = semantic_threshold
        
        # Cloud vector store configuration
        if vector_store == 'qdrant_cloud':
            config.qdrant_url = qdrant_url
            config.qdrant_api_key = qdrant_api_key
            config.qdrant_collection_name = qdrant_collection
            config.qdrant_prefer_grpc = qdrant_grpc
        
        # Save updated config
        import json
        config_path = rag_manager.config_dir / f"{name}.json"
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        click.echo(f"✅ Knowledge base '{name}' created successfully!")
        
        if verbose:
            click.echo(f"📋 Configuration:")
            click.echo(f"   Embedding model: {config.embedding_model}")
            click.echo(f"   Vector store: {config.vector_store}")
            click.echo(f"   Chunking strategy: {config.chunking_strategy}")
            click.echo(f"   Chunk size: {config.chunk_size}")
            click.echo(f"   Chunk overlap: {config.chunk_overlap}")
            
            if vector_store == 'qdrant_cloud':
                click.echo(f"   🌐 Qdrant Cloud URL: {qdrant_url}")
                if qdrant_collection:
                    click.echo(f"   📦 Collection: {qdrant_collection}")
                if qdrant_grpc:
                    click.echo(f"   ⚡ gRPC enabled for better performance")
            
            if config.use_hybrid_search:
                click.echo(f"   🔄 Hybrid search enabled (α={config.hybrid_alpha})")
            if config.use_reranking:
                click.echo(f"   🎯 Reranking enabled ({config.reranker_model})")
            if config.chunking_strategy == 'semantic':
                click.echo(f"   🧠 Semantic chunking (threshold={config.semantic_similarity_threshold})")
        
        # Show cloud setup info
        if vector_store == 'qdrant_cloud':
            click.echo()
            click.echo("🌐 Enterprise Cloud Setup Complete!")
            click.echo("📊 Your data will be stored securely in Qdrant Cloud")
            click.echo("🔒 Benefits: High availability, scalability, and enterprise security")
            click.echo("📈 Perfect for production deployments and team collaboration")
        
    except Exception as e:
        click.echo(f"❌ Error creating knowledge base: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@rag.command('list-kb')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def list_kb(verbose):
    """List all knowledge bases"""
    
    try:
        rag_manager = RAGManager(verbose=verbose)
        kbs = rag_manager.list_knowledge_bases()
        
        if not kbs:
            click.echo("📭 No knowledge bases found")
            return
        
        click.echo("📚 Knowledge Bases:")
        click.echo("=" * 60)
        
        for kb in kbs:
            click.echo(f"🔹 {kb.name}")
            if kb.description:
                click.echo(f"   Description: {kb.description}")
            click.echo(f"   Documents: {kb.document_count}")
            click.echo(f"   Embedding Model: {kb.embedding_model}")
            click.echo(f"   Created: {kb.created_at}")
            if verbose:
                click.echo(f"   Chunk Size: {kb.chunk_size}")
                click.echo(f"   Chunk Overlap: {kb.chunk_overlap}")
                click.echo(f"   Strategy: {kb.chunking_strategy}")
            click.echo()
        
    except Exception as e:
        click.echo(f"❌ Error listing knowledge bases: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@rag.command('delete-kb')
@click.option('--name', '-n', required=True, help='Knowledge base name')
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def delete_kb(name, confirm, verbose):
    """Delete a knowledge base"""
    
    if not confirm:
        if not click.confirm(f"Are you sure you want to delete knowledge base '{name}'?"):
            click.echo("❌ Operation cancelled")
            return
    
    try:
        rag_manager = RAGManager(verbose=verbose)
        success = rag_manager.delete_knowledge_base(name)
        
        if success:
            click.echo(f"✅ Knowledge base '{name}' deleted successfully!")
        else:
            click.echo(f"❌ Knowledge base '{name}' not found", err=True)
            sys.exit(1)
        
    except Exception as e:
        click.echo(f"❌ Error deleting knowledge base: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@rag.command('ingest')
@click.option('--kb', '-k', required=True, help='Knowledge base name')
@click.option('--source', '-s', required=True, help='Source path (file or directory)')
@click.option('--file-types', '-t', default='pdf,txt,md,docx', 
              help='Comma-separated list of file types to include')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def ingest(kb, source, file_types, verbose):
    """Ingest documents into a knowledge base"""
    
    try:
        rag_manager = RAGManager(verbose=verbose)
        
        # Parse file types
        file_types_list = [ft.strip() for ft in file_types.split(',')]
        
        click.echo(f"📥 Ingesting documents from: {source}")
        click.echo(f"📁 Into knowledge base: {kb}")
        click.echo(f"📋 File types: {', '.join(file_types_list)}")
        
        result = rag_manager.ingest_documents(
            kb_name=kb,
            source_path=source,
            file_types=file_types_list
        )
        
        click.echo()
        click.echo("📊 Ingestion Summary:")
        click.echo(f"   Documents processed: {result['documents_processed']}")
        click.echo(f"   Chunks created: {result['chunks_created']}")
        click.echo(f"   Total documents in KB: {result['total_documents']}")
        
        if result['documents_processed'] > 0:
            click.echo("✅ Document ingestion completed successfully!")
        else:
            click.echo("⚠️  No documents were processed")
        
    except Exception as e:
        click.echo(f"❌ Error ingesting documents: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@rag.command('query')
@click.option('--kb', '-k', required=True, help='Knowledge base name')
@click.option('--question', '-q', required=True, help='Question to ask')
@click.option('--top-k', '-t', type=int, default=5, help='Number of documents to retrieve')
@click.option('--llm', '-l', help='LLM model for answer generation (optional)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def query_kb(kb, question, top_k, llm, verbose):
    """Query a knowledge base"""
    
    try:
        rag_manager = RAGManager(verbose=verbose)
        
        click.echo(f"🔍 Querying knowledge base: {kb}")
        click.echo(f"❓ Question: {question}")
        
        result = rag_manager.query(
            kb_name=kb,
            question=question,
            top_k=top_k,
            llm_model=llm
        )
        
        click.echo()
        click.echo("📋 Retrieved Documents:")
        click.echo("=" * 60)
        
        for doc in result['retrieved_documents']:
            click.echo(f"📄 Rank {doc['rank']}:")
            click.echo(f"   Content: {doc['content'][:200]}...")
            if verbose and doc['metadata']:
                click.echo(f"   Metadata: {doc['metadata']}")
            click.echo()
        
        if result['answer']:
            click.echo("💡 Generated Answer:")
            click.echo("=" * 60)
            click.echo(result['answer'])
        elif llm:
            click.echo("⚠️  Could not generate answer with LLM")
        
    except Exception as e:
        click.echo(f"❌ Error querying knowledge base: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@rag.command('stats')
@click.option('--kb', '-k', required=True, help='Knowledge base name')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def kb_stats(kb, verbose):
    """Show knowledge base statistics"""
    
    try:
        rag_manager = RAGManager(verbose=verbose)
        stats = rag_manager.get_stats(kb)
        
        click.echo(f"📊 Knowledge Base Statistics: {kb}")
        click.echo("=" * 60)
        click.echo(f"Name: {stats['name']}")
        click.echo(f"Description: {stats['description']}")
        click.echo(f"Embedding Model: {stats['embedding_model']}")
        click.echo(f"Vector Store: {stats['vector_store']}")
        click.echo(f"Document Count: {stats['document_count']}")
        click.echo(f"Chunk Count: {stats['chunk_count']}")
        click.echo(f"Chunk Size: {stats['chunk_size']}")
        click.echo(f"Chunk Overlap: {stats['chunk_overlap']}")
        click.echo(f"Chunking Strategy: {stats['chunking_strategy']}")
        click.echo(f"Created: {stats['created_at']}")
        click.echo(f"Updated: {stats['updated_at']}")
        
    except Exception as e:
        click.echo(f"❌ Error getting knowledge base stats: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@rag.command('evaluate')
@click.option('--kb', '-k', required=True, help='Knowledge base name')
@click.option('--questions-file', '-f', help='JSON file with test questions and ground truths')
@click.option('--questions', '-q', help='Comma-separated list of test questions')
@click.option('--num-auto-questions', '-n', type=int, default=5, 
              help='Number of auto-generated questions (if no questions provided)')
@click.option('--llm-model', '-l', help='LLM model for answer generation')
@click.option('--local-llm', help='Local LLM model to use (e.g., llama2, mistral)')
@click.option('--openai-key', help='OpenAI API key for RAGAS evaluation')
@click.option('--metrics', '-m', default='faithfulness,answer_relevancy',
              help='Comma-separated list of RAGAS metrics')
@click.option('--output-dir', '-o', default='./rag_evaluations', help='Output directory for results')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def evaluate_rag(kb, questions_file, questions, num_auto_questions, llm_model, 
                 local_llm, openai_key, metrics, output_dir, verbose):
    """Evaluate RAG pipeline using RAGAS framework"""
    
    try:
        # Initialize RAG manager and evaluator
        rag_manager = RAGManager(verbose=verbose)
        evaluator = RAGEvaluator(rag_manager, verbose=verbose)
        
        # Prepare test questions
        test_questions = []
        ground_truths = None
        
        if questions_file:
            # Load from file
            test_questions, ground_truths = evaluator.load_test_questions_from_file(questions_file)
            click.echo(f"📄 Loaded {len(test_questions)} questions from {questions_file}")
            
        elif questions:
            # Use provided questions
            test_questions = [q.strip() for q in questions.split(',')]
            click.echo(f"❓ Using {len(test_questions)} provided questions")
            
        else:
            # Auto-generate questions
            click.echo(f"🤖 Auto-generating {num_auto_questions} test questions")
            test_questions = evaluator.generate_test_questions(kb, num_auto_questions)
        
        if not test_questions:
            click.echo("❌ No test questions available", err=True)
            sys.exit(1)
        
        # Parse metrics
        metric_list = [m.strip() for m in metrics.split(',')]
        
        # Create evaluation config
        config = RAGEvaluationConfig(
            kb_name=kb,
            test_questions=test_questions,
            ground_truths=ground_truths,
            llm_model=llm_model,
            local_llm=local_llm,
            openai_key=openai_key,
            output_dir=output_dir,
            metrics=metric_list
        )
        
        click.echo(f"🧪 Starting RAGAS evaluation for KB: {kb}")
        click.echo(f"📊 Metrics: {', '.join(metric_list)}")
        
        # Run evaluation
        result = evaluator.evaluate_rag_pipeline(config)
        
        # Display results
        click.echo("\n" + "="*60)
        click.echo("🎯 RAGAS Evaluation Results")
        click.echo("="*60)
        
        for metric, score in result.metrics_scores.items():
            performance = evaluator._get_performance_label(score)
            click.echo(f"{metric.replace('_', ' ').title()}: {score:.3f} {performance}")
        
        avg_score = sum(result.metrics_scores.values()) / len(result.metrics_scores)
        overall_performance = evaluator._get_performance_label(avg_score)
        click.echo(f"\n📈 Overall Score: {avg_score:.3f} {overall_performance}")
        
        click.echo(f"\n💾 Detailed results saved to: rag_evaluations/{result.evaluation_id}/")
        click.echo(f"📄 Summary report: rag_evaluations/{result.evaluation_id}/summary_report.md")
        
    except Exception as e:
        click.echo(f"❌ RAG evaluation failed: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@rag.command('compare-kb')
@click.option('--kbs', '-k', required=True, help='Comma-separated list of knowledge base names')
@click.option('--questions-file', '-f', help='JSON file with test questions')
@click.option('--num-questions', '-n', type=int, default=5, 
              help='Number of test questions for comparison')
@click.option('--llm-model', '-l', help='LLM model for answer generation')
@click.option('--metrics', '-m', default='faithfulness,answer_relevancy',
              help='Comma-separated list of RAGAS metrics')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def compare_knowledge_bases(kbs, questions_file, num_questions, llm_model, metrics, verbose):
    """Compare multiple knowledge bases using RAGAS evaluation"""
    
    try:
        kb_list = [kb.strip() for kb in kbs.split(',')]
        
        if len(kb_list) < 2:
            click.echo("❌ At least 2 knowledge bases required for comparison", err=True)
            sys.exit(1)
        
        # Initialize evaluator
        rag_manager = RAGManager(verbose=verbose)
        evaluator = RAGEvaluator(rag_manager, verbose=verbose)
        
        # Prepare test questions
        if questions_file:
            test_questions, ground_truths = evaluator.load_test_questions_from_file(questions_file)
        else:
            # Use the first KB to generate questions
            test_questions = evaluator.generate_test_questions(kb_list[0], num_questions)
            ground_truths = None
        
        click.echo(f"🔄 Comparing {len(kb_list)} knowledge bases")
        click.echo(f"❓ Using {len(test_questions)} test questions")
        
        # Evaluate each knowledge base
        results = {}
        metric_list = [m.strip() for m in metrics.split(',')]
        
        for kb_name in kb_list:
            click.echo(f"\n🧪 Evaluating: {kb_name}")
            
            config = RAGEvaluationConfig(
                kb_name=kb_name,
                test_questions=test_questions,
                ground_truths=ground_truths,
                llm_model=llm_model,
                metrics=metric_list
            )
            
            result = evaluator.evaluate_rag_pipeline(config)
            results[kb_name] = result.metrics_scores
        
        # Display comparison
        click.echo("\n" + "="*80)
        click.echo("📊 Knowledge Base Comparison Results")
        click.echo("="*80)
        
        # Create comparison table
        import pandas as pd
        df = pd.DataFrame(results).T  # Transpose for KB names as rows
        
        click.echo(f"\n{df.round(3).to_string()}")
        
        # Find best performing KB for each metric
        click.echo(f"\n🏆 Best Performers:")
        for metric in metric_list:
            if metric in df.columns:
                best_kb = df[metric].idxmax()
                best_score = df[metric].max()
                click.echo(f"   {metric.replace('_', ' ').title()}: {best_kb} ({best_score:.3f})")
        
        # Overall best
        df['average'] = df.mean(axis=1)
        overall_best = df['average'].idxmax()
        overall_score = df['average'].max()
        
        click.echo(f"\n🎯 Overall Best: {overall_best} (avg: {overall_score:.3f})")
        
    except Exception as e:
        click.echo(f"❌ KB comparison failed: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@rag.command('generate-test-questions')
@click.option('--kb', '-k', required=True, help='Knowledge base name')
@click.option('--num-questions', '-n', type=int, default=10, help='Number of questions to generate')
@click.option('--output-file', '-o', help='Output JSON file for questions')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def generate_test_questions(kb, num_questions, output_file, verbose):
    """Generate test questions for RAG evaluation"""
    
    try:
        rag_manager = RAGManager(verbose=verbose)
        evaluator = RAGEvaluator(rag_manager, verbose=verbose)
        
        click.echo(f"🤖 Generating {num_questions} test questions for KB: {kb}")
        
        questions = evaluator.generate_test_questions(kb, num_questions)
        
        # Prepare output data
        output_data = {
            "kb_name": kb,
            "questions": questions,
            "ground_truths": None,  # User can fill these in manually
            "metadata": {
                "generated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
                "num_questions": len(questions)
            }
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            click.echo(f"💾 Questions saved to: {output_file}")
        else:
            click.echo(f"\n📝 Generated Questions:")
            for i, question in enumerate(questions, 1):
                click.echo(f"   {i}. {question}")
        
        click.echo(f"\n💡 Tip: Add ground truth answers to the JSON file for context_recall evaluation")
        
    except Exception as e:
        click.echo(f"❌ Question generation failed: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@rag.command('agentic-query')
@click.option('--kb', '-k', required=True, help='Knowledge base name')
@click.option('--question', '-q', required=True, help='Question to ask')
@click.option('--llm-model', '-l', default='openai:gpt-4', help='LLM model for agentic reasoning')
@click.option('--temperature', '-t', type=float, default=0.0, help='LLM temperature')
@click.option('--show-trace', is_flag=True, help='Show workflow trace')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def agentic_query(kb, question, llm_model, temperature, show_trace, verbose):
    """Query knowledge base using Agentic RAG with intelligent decision making"""
    
    try:
        # Initialize traditional RAG manager
        rag_manager = RAGManager(verbose=verbose)
        
        # Initialize Agentic RAG manager
        agentic_config = AgenticRAGConfig(
            kb_name=kb,
            llm_model=llm_model,
            temperature=temperature,
            verbose=verbose
        )
        
        agentic_rag = AgenticRAGManager(rag_manager, agentic_config)
        
        click.echo(f"🤖 Agentic RAG Query")
        click.echo(f"Knowledge Base: {kb}")
        click.echo(f"Question: {question}")
        click.echo("=" * 60)
        
        # Query using Agentic RAG
        result = agentic_rag.query(question)
        
        # Display results
        click.echo("\n💡 Agentic RAG Answer:")
        click.echo("=" * 60)
        click.echo(result['answer'])
        
        if show_trace and result['workflow_trace']:
            click.echo(f"\n🔄 Workflow Trace:")
            for step in result['workflow_trace']:
                click.echo(f"   {step}")
        
        click.echo(f"\n📊 Messages processed: {result['message_count']}")
        
    except Exception as e:
        click.echo(f"❌ Agentic RAG query failed: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@rag.command('compare-rag-types')
@click.option('--kb', '-k', required=True, help='Knowledge base name')
@click.option('--question', '-q', required=True, help='Question to ask')
@click.option('--llm-model', '-l', default='openai:gpt-4', help='LLM model for agentic RAG')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def compare_rag_types(kb, question, llm_model, verbose):
    """Compare Traditional RAG vs Agentic RAG side by side"""
    
    try:
        click.echo(f"🆚 RAG Comparison: Traditional vs Agentic")
        click.echo(f"Knowledge Base: {kb}")
        click.echo(f"Question: {question}")
        click.echo("=" * 80)
        
        # Initialize RAG manager
        rag_manager = RAGManager(verbose=False)  # Reduce noise
        
        # Test Traditional RAG
        click.echo("\n🔹 Traditional RAG:")
        click.echo("-" * 40)
        
        import time
        start_time = time.time()
        
        traditional_result = rag_manager.query(
            kb_name=kb,
            question=question,
            top_k=5,
            llm_model=llm_model if llm_model != "openai:gpt-4" else None
        )
        
        traditional_time = time.time() - start_time
        
        if traditional_result.get('answer'):
            click.echo(traditional_result['answer'])
        else:
            click.echo("Retrieved documents:")
            for i, doc in enumerate(traditional_result['retrieved_documents'][:2], 1):
                click.echo(f"   {i}. {doc['content'][:100]}...")
        
        # Test Agentic RAG
        click.echo(f"\n🤖 Agentic RAG:")
        click.echo("-" * 40)
        
        start_time = time.time()
        
        agentic_config = AgenticRAGConfig(
            kb_name=kb,
            llm_model=llm_model,
            verbose=False
        )
        
        agentic_rag = AgenticRAGManager(rag_manager, agentic_config)
        agentic_result = agentic_rag.query(question)
        
        agentic_time = time.time() - start_time
        
        click.echo(agentic_result['answer'])
        
        # Comparison summary
        click.echo(f"\n📊 Comparison Summary:")
        click.echo("=" * 80)
        click.echo(f"⏱️  Traditional RAG time: {traditional_time:.2f}s")
        click.echo(f"⏱️  Agentic RAG time: {agentic_time:.2f}s")
        click.echo(f"🔄 Agentic workflow steps: {len(agentic_result['workflow_trace'])}")
        
        # Workflow trace for agentic
        if agentic_result['workflow_trace']:
            click.echo(f"\n🤖 Agentic RAG Process:")
            for step in agentic_result['workflow_trace']:
                click.echo(f"   {step}")
        
        click.echo(f"\n💡 Key Differences:")
        click.echo("   🔹 Traditional: Direct retrieval → generation")
        click.echo("   🤖 Agentic: Intelligent decisions → document grading → adaptive response")
        
    except Exception as e:
        click.echo(f"❌ RAG comparison failed: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@rag.command('show-workflow')
@click.option('--type', '-t', type=click.Choice(['traditional', 'agentic']), 
              default='agentic', help='RAG type to visualize')
def show_workflow(type):
    """Show RAG workflow visualization"""
    
    if type == 'traditional':
        click.echo("🔹 Traditional RAG Workflow:")
        click.echo("=" * 40)
        click.echo("""
[User Query] → [Retrieve Documents] → [Generate Answer] → [Response]
             ↓                      ↓
        [Vector Search]         [LLM with Context]
        """)
        
        click.echo("\n📋 Process:")
        click.echo("1. User asks a question")
        click.echo("2. System retrieves relevant documents")
        click.echo("3. LLM generates answer using retrieved context")
        click.echo("4. Response is returned to user")
        
    else:  # agentic
        click.echo("🤖 Agentic RAG Workflow:")
        click.echo("=" * 40)
        click.echo("""
[User Query] → [Agent Decision]
                      ↓
              [Need Retrieval?] → [Direct Response] → [END]
                      ↓
              [Retrieve Docs] → [Grade Relevance]
                      ↓              ↓
              [Relevant?] → [Generate Answer] → [END]
                      ↓
              [Rewrite Query] → [Agent Decision]
        """)
        
        click.echo("\n📋 Enhanced Process:")
        click.echo("1. 🧠 Agent analyzes if retrieval is needed")
        click.echo("2. 🔍 If needed, retrieves relevant documents")
        click.echo("3. 📊 Grades document relevance to question")
        click.echo("4. ✍️  Rewrites question if documents aren't relevant")
        click.echo("5. 🔄 Loops until relevant docs found")
        click.echo("6. 📝 Generates final answer with best context")
        
        click.echo(f"\n💡 Advantages:")
        click.echo("   • Intelligent retrieval decisions")
        click.echo("   • Document relevance grading")
        click.echo("   • Adaptive question rewriting")
        click.echo("   • Higher quality responses")

# Add Fine-tuning Group
@cli.group()
def finetune():
    """Supervised fine-tuning commands with LoRA, QLoRA, and full fine-tuning support"""
    pass

@finetune.command('instruct')
@click.option('--model', '-m', required=True, help='Base model name (e.g., microsoft/DialoGPT-medium, meta-llama/Llama-2-7b-hf)')
@click.option('--dataset', '-d', required=True, help='Path to dataset file (JSON, JSONL, CSV)')
@click.option('--output-dir', '-o', default='./fine_tuned_model', help='Output directory for fine-tuned model')
@click.option('--method', type=click.Choice(['lora', 'qlora', 'full']), default='lora', 
              help='Fine-tuning method')
@click.option('--task-type', type=click.Choice(['instruction_following', 'chat', 'completion', 'classification']),
              default='instruction_following', help='Task type for fine-tuning')
@click.option('--dataset-format', type=click.Choice(['instruction', 'chat', 'completion', 'classification']),
              default='instruction', help='Dataset format')
@click.option('--epochs', type=int, default=3, help='Number of training epochs')
@click.option('--batch-size', type=int, default=4, help='Training batch size per device')
@click.option('--learning-rate', type=float, default=2e-4, help='Learning rate')
@click.option('--lora-rank', type=int, default=16, help='LoRA rank (only for LoRA/QLoRA)')
@click.option('--lora-alpha', type=int, default=32, help='LoRA alpha (only for LoRA/QLoRA)')
@click.option('--lora-dropout', type=float, default=0.1, help='LoRA dropout (only for LoRA/QLoRA)')
@click.option('--lora-target-modules', help='Comma-separated LoRA target modules (auto-detect if not specified)')
@click.option('--load-in-4bit', is_flag=True, help='Use 4-bit quantization (QLoRA)')
@click.option('--load-in-8bit', is_flag=True, help='Use 8-bit quantization')
@click.option('--max-seq-length', type=int, default=512, help='Maximum sequence length')
@click.option('--gradient-accumulation-steps', type=int, default=1, help='Gradient accumulation steps')
@click.option('--warmup-ratio', type=float, default=0.03, help='Warmup ratio')
@click.option('--weight-decay', type=float, default=0.01, help='Weight decay')
@click.option('--lr-scheduler', default='cosine', help='Learning rate scheduler type')
@click.option('--eval-steps', type=int, default=500, help='Evaluation frequency in steps')
@click.option('--save-steps', type=int, default=500, help='Save frequency in steps')
@click.option('--logging-steps', type=int, default=10, help='Logging frequency in steps')
@click.option('--test-size', type=float, default=0.1, help='Fraction of data for evaluation')
@click.option('--fp16', is_flag=True, help='Use mixed precision training (FP16)')
@click.option('--bf16', is_flag=True, help='Use bfloat16 precision')
@click.option('--gradient-checkpointing', is_flag=True, help='Enable gradient checkpointing')
@click.option('--packing', is_flag=True, help='Enable sequence packing (for SFTTrainer)')
@click.option('--report-to', multiple=True, default=['tensorboard'], 
              help='Experiment tracking platforms (tensorboard, wandb, comet)')
@click.option('--wandb-project', help='Weights & Biases project name')
@click.option('--wandb-entity', help='Weights & Biases entity name')
@click.option('--run-name', help='Name for this training run')
@click.option('--push-to-hub', is_flag=True, help='Push model to Hugging Face Hub after training')
@click.option('--hub-model-id', help='Hugging Face Hub model ID (username/model-name)')
@click.option('--hf-token', help='Hugging Face token')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--monitor/--no-monitor', default=True, help='Enable Langfuse monitoring')
def finetune_instruct(model, dataset, output_dir, method, task_type, dataset_format, epochs, 
                     batch_size, learning_rate, lora_rank, lora_alpha, lora_dropout, 
                     lora_target_modules, load_in_4bit, load_in_8bit, max_seq_length,
                     gradient_accumulation_steps, warmup_ratio, weight_decay, lr_scheduler,
                     eval_steps, save_steps, logging_steps, test_size, fp16, bf16, 
                     gradient_checkpointing, packing, report_to, wandb_project, wandb_entity,
                     run_name, push_to_hub, hub_model_id, hf_token, verbose, monitor):
    """Fine-tune models for instruction following, chat, and other tasks"""
    
    if verbose:
        click.echo("🚀 Starting supervised fine-tuning...")
        click.echo(f"   Model: {model}")
        click.echo(f"   Method: {method}")
        click.echo(f"   Dataset: {dataset}")
        click.echo(f"   Task: {task_type}")
    
    # Parse target modules
    target_modules = None
    if lora_target_modules:
        target_modules = [m.strip() for m in lora_target_modules.split(',')]
    
    # Setup Weights & Biases if specified
    report_to_list = list(report_to) if report_to else ["tensorboard"]
    if wandb_project and "wandb" not in report_to_list:
        report_to_list.append("wandb")
    
    # Set environment variables for tracking
    if wandb_project:
        os.environ['WANDB_PROJECT'] = wandb_project
    if wandb_entity:
        os.environ['WANDB_ENTITY'] = wandb_entity
    if run_name:
        os.environ['WANDB_RUN_NAME'] = run_name
    
    # Create fine-tuning configuration
    config = create_fine_tuning_config(
        model_name=model,
        method=method,
        task_type=task_type,
        dataset_format=dataset_format,
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=target_modules,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        max_seq_length=max_seq_length,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler,
        eval_steps=eval_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=gradient_checkpointing,
        packing=packing,
        report_to=report_to_list,
    )
    
    try:
        # Initialize fine-tuner
        fine_tuner = SupervisedFineTuner(config, verbose=verbose)
        
        # Setup model and tokenizer
        fine_tuner.setup_model_and_tokenizer()
        
        # Prepare dataset
        train_dataset, eval_dataset = fine_tuner.prepare_dataset(dataset, test_size=test_size)
        
        # Setup trainer
        fine_tuner.setup_trainer(train_dataset, eval_dataset)
        
        # Start training
        result = fine_tuner.train()
        
        # Evaluate if eval dataset exists
        if eval_dataset:
            eval_results = fine_tuner.evaluate()
            if verbose:
                click.echo("📊 Final evaluation results:")
                for key, value in eval_results.items():
                    click.echo(f"   {key}: {value:.4f}")
        
        # Push to Hub if requested
        if push_to_hub:
            if not hub_model_id:
                click.echo("❌ --hub-model-id required when using --push-to-hub", err=True)
                sys.exit(1)
            
            if hf_token:
                os.environ['HF_TOKEN'] = hf_token
            
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                
                if method in ["lora", "qlora"]:
                    # Push LoRA adapters
                    api.upload_folder(
                        folder_path=output_dir,
                        repo_id=hub_model_id,
                        repo_type="model",
                        token=hf_token
                    )
                else:
                    # Push full model
                    fine_tuner.model.push_to_hub(hub_model_id, token=hf_token)
                    fine_tuner.tokenizer.push_to_hub(hub_model_id, token=hf_token)
                
                click.echo(f"✅ Model pushed to Hub: https://huggingface.co/{hub_model_id}")
            except Exception as e:
                click.echo(f"❌ Failed to push to Hub: {e}")
        
        click.echo("✅ Fine-tuning completed successfully!")
        click.echo(f"📁 Model saved to: {output_dir}")
        
    except Exception as e:
        click.echo(f"❌ Fine-tuning failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@finetune.command('lora')
@click.option('--model', '-m', required=True, help='Base model name')
@click.option('--dataset', '-d', required=True, help='Path to dataset file')
@click.option('--output-dir', '-o', default='./lora_model', help='Output directory')
@click.option('--rank', '-r', type=int, default=16, help='LoRA rank')
@click.option('--alpha', '-a', type=int, default=32, help='LoRA alpha')
@click.option('--dropout', type=float, default=0.1, help='LoRA dropout')
@click.option('--target-modules', help='Comma-separated target modules')
@click.option('--epochs', type=int, default=3, help='Number of epochs')
@click.option('--batch-size', type=int, default=4, help='Batch size')
@click.option('--learning-rate', type=float, default=2e-4, help='Learning rate')
@click.option('--max-seq-length', type=int, default=512, help='Max sequence length')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def finetune_lora(model, dataset, output_dir, rank, alpha, dropout, target_modules, 
                  epochs, batch_size, learning_rate, max_seq_length, verbose):
    """LoRA fine-tuning with simplified options"""
    
    target_modules_list = None
    if target_modules:
        target_modules_list = [m.strip() for m in target_modules.split(',')]
    
    config = create_fine_tuning_config(
        model_name=model,
        method="lora",
        output_dir=output_dir,
        lora_rank=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        lora_target_modules=target_modules_list,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
    )
    
    try:
        fine_tuner = SupervisedFineTuner(config, verbose=verbose)
        fine_tuner.setup_model_and_tokenizer()
        
        train_dataset, eval_dataset = fine_tuner.prepare_dataset(dataset)
        fine_tuner.setup_trainer(train_dataset, eval_dataset)
        
        result = fine_tuner.train()
        
        if eval_dataset:
            fine_tuner.evaluate()
        
        click.echo("✅ LoRA fine-tuning completed!")
        click.echo(f"📁 Model saved to: {output_dir}")
        
    except Exception as e:
        click.echo(f"❌ LoRA fine-tuning failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@finetune.command('qlora')
@click.option('--model', '-m', required=True, help='Base model name')
@click.option('--dataset', '-d', required=True, help='Path to dataset file')
@click.option('--output-dir', '-o', default='./qlora_model', help='Output directory')
@click.option('--rank', '-r', type=int, default=64, help='LoRA rank for QLoRA')
@click.option('--alpha', '-a', type=int, default=16, help='LoRA alpha for QLoRA')
@click.option('--epochs', type=int, default=3, help='Number of epochs')
@click.option('--batch-size', type=int, default=1, help='Batch size (usually small for QLoRA)')
@click.option('--gradient-accumulation-steps', type=int, default=4, help='Gradient accumulation steps')
@click.option('--learning-rate', type=float, default=2e-4, help='Learning rate')
@click.option('--max-seq-length', type=int, default=512, help='Max sequence length')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def finetune_qlora(model, dataset, output_dir, rank, alpha, epochs, batch_size, 
                   gradient_accumulation_steps, learning_rate, max_seq_length, verbose):
    """QLoRA fine-tuning with 4-bit quantization"""
    
    config = create_fine_tuning_config(
        model_name=model,
        method="qlora",
        output_dir=output_dir,
        lora_rank=rank,
        lora_alpha=alpha,
        load_in_4bit=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
    )
    
    try:
        fine_tuner = SupervisedFineTuner(config, verbose=verbose)
        fine_tuner.setup_model_and_tokenizer()
        
        train_dataset, eval_dataset = fine_tuner.prepare_dataset(dataset)
        fine_tuner.setup_trainer(train_dataset, eval_dataset)
        
        result = fine_tuner.train()
        
        if eval_dataset:
            fine_tuner.evaluate()
        
        click.echo("✅ QLoRA fine-tuning completed!")
        click.echo(f"📁 Model saved to: {output_dir}")
        
    except Exception as e:
        click.echo(f"❌ QLoRA fine-tuning failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@finetune.command('full')
@click.option('--model', '-m', required=True, help='Base model name')
@click.option('--dataset', '-d', required=True, help='Path to dataset file')
@click.option('--output-dir', '-o', default='./full_finetuned_model', help='Output directory')
@click.option('--epochs', type=int, default=3, help='Number of epochs')
@click.option('--batch-size', type=int, default=2, help='Batch size (usually small for full fine-tuning)')
@click.option('--gradient-accumulation-steps', type=int, default=2, help='Gradient accumulation steps')
@click.option('--learning-rate', type=float, default=5e-5, help='Learning rate (lower for full fine-tuning)')
@click.option('--max-seq-length', type=int, default=512, help='Max sequence length')
@click.option('--gradient-checkpointing', is_flag=True, help='Enable gradient checkpointing')
@click.option('--fp16', is_flag=True, help='Use FP16 mixed precision')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def finetune_full(model, dataset, output_dir, epochs, batch_size, gradient_accumulation_steps,
                  learning_rate, max_seq_length, gradient_checkpointing, fp16, verbose):
    """Full model fine-tuning"""
    
    config = create_fine_tuning_config(
        model_name=model,
        method="full",
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        gradient_checkpointing=gradient_checkpointing,
        fp16=fp16,
    )
    
    try:
        fine_tuner = SupervisedFineTuner(config, verbose=verbose)
        fine_tuner.setup_model_and_tokenizer()
        
        train_dataset, eval_dataset = fine_tuner.prepare_dataset(dataset)
        fine_tuner.setup_trainer(train_dataset, eval_dataset)
        
        result = fine_tuner.train()
        
        if eval_dataset:
            fine_tuner.evaluate()
        
        click.echo("✅ Full fine-tuning completed!")
        click.echo(f"📁 Model saved to: {output_dir}")
        
    except Exception as e:
        click.echo(f"❌ Full fine-tuning failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@finetune.command('evaluate')
@click.option('--model-path', '-m', required=True, help='Path to fine-tuned model')
@click.option('--base-model', help='Base model name (for LoRA/QLoRA models)')
@click.option('--test-dataset', '-d', help='Path to test dataset')
@click.option('--prompt', '-p', help='Single prompt to test')
@click.option('--max-length', type=int, default=512, help='Maximum generation length')
@click.option('--temperature', type=float, default=0.7, help='Generation temperature')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def evaluate_finetuned(model_path, base_model, test_dataset, prompt, max_length, temperature, verbose):
    """Evaluate or test a fine-tuned model"""
    
    try:
        if verbose:
            click.echo(f"🔍 Loading fine-tuned model from: {model_path}")
        
        # Load model
        model, tokenizer, config = SupervisedFineTuner.load_fine_tuned_model(model_path, base_model)
        
        if prompt:
            # Single prompt testing
            if verbose:
                click.echo(f"💬 Testing prompt: {prompt}")
            
            # Format prompt based on task type
            if config.task_type == "instruction_following":
                formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
            elif config.task_type == "chat":
                formatted_prompt = f"Human: {prompt}\n\nAssistant: "
            else:
                formatted_prompt = prompt
            
            # Generate response
            inputs = tokenizer(formatted_prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the generated part
            generated = response[len(formatted_prompt):].strip()
            
            click.echo("\n" + "="*50)
            click.echo("🤖 Generated Response:")
            click.echo("="*50)
            click.echo(generated)
            click.echo("="*50)
        
        elif test_dataset:
            # Dataset evaluation
            if verbose:
                click.echo(f"📊 Evaluating on dataset: {test_dataset}")
            
            # TODO: Implement dataset evaluation
            click.echo("📊 Dataset evaluation not yet implemented")
        
        else:
            click.echo("❌ Either --prompt or --test-dataset must be provided")
            sys.exit(1)
    
    except Exception as e:
        click.echo(f"❌ Evaluation failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@finetune.command('list-models')
@click.option('--path', '-p', default='.', help='Directory to search for fine-tuned models')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def list_finetuned_models(path, verbose):
    """List available fine-tuned models"""
    
    import glob
    
    model_dirs = []
    
    # Search for model directories
    for pattern in ['**/fine_tuning_config.json', '**/adapter_config.json', '**/config.json']:
        for config_file in glob.glob(os.path.join(path, pattern), recursive=True):
            model_dir = os.path.dirname(config_file)
            if model_dir not in model_dirs:
                model_dirs.append(model_dir)
    
    if not model_dirs:
        click.echo("❌ No fine-tuned models found")
        return
    
    click.echo("📁 Fine-tuned models found:")
    click.echo("="*50)
    
    for model_dir in sorted(model_dirs):
        rel_path = os.path.relpath(model_dir, path)
        
        # Try to load config info
        config_info = "Unknown configuration"
        try:
            ft_config_path = os.path.join(model_dir, "fine_tuning_config.json")
            if os.path.exists(ft_config_path):
                with open(ft_config_path, 'r') as f:
                    config = json.load(f)
                    method = config.get('method', 'unknown')
                    model_name = config.get('model_name', 'unknown')
                    config_info = f"{method} fine-tuning of {model_name}"
        except:
            pass
        
        click.echo(f"📂 {rel_path}")
        if verbose:
            click.echo(f"   {config_info}")
        click.echo()

# Add MCP Commands Group
@cli.group()
def mcp():
    """Model Context Protocol (MCP) integration commands"""
    pass

@mcp.command('list-servers')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed server information')
def list_mcp_servers(verbose):
    """List all available MCP servers"""
    try:
        manager = get_mcp_manager(verbose=verbose)
        servers = manager.list_available_servers()
        
        if not servers:
            click.echo("No MCP servers available.")
            return
        
        click.echo(f"📡 Available MCP Servers ({len(servers)} total):\n")
        
        for server in servers:
            status_icon = "🟢" if server["status"] == "connected" else "⚪"
            auth_icon = "🔐" if server.get("requires_auth", False) else "🔓"
            
            click.echo(f"{status_icon} {auth_icon} {server['display_name']} ({server['name']})")
            click.echo(f"   Description: {server['description']}")
            
            if verbose:
                click.echo(f"   Type: {server['type']}")
                click.echo(f"   Command: {server['command']} {' '.join(server['args'])}")
                click.echo(f"   Status: {server['status']}")
                if server.get("requires_auth"):
                    click.echo(f"   Auth Required: {server['auth_instructions']}")
            
            click.echo()
    
    except Exception as e:
        click.echo(f"❌ Error listing servers: {e}", err=True)
        sys.exit(1)

@mcp.command('server-info')
@click.argument('server_name')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed information')
def show_server_info(server_name, verbose):
    """Show detailed information about a specific MCP server"""
    try:
        manager = get_mcp_manager(verbose=verbose)
        info = manager.get_server_info(server_name)
        
        if not info:
            click.echo(f"❌ Server '{server_name}' not found.", err=True)
            sys.exit(1)
        
        click.echo(f"📡 MCP Server: {info['display_name']} ({info['name']})")
        click.echo(f"Description: {info['description']}")
        click.echo(f"Type: {info['type']}")
        click.echo(f"Status: {info['status']}")
        click.echo(f"Command: {info['command']} {' '.join(info['args'])}")
        
        if info.get("requires_auth"):
            click.echo(f"🔐 Authentication Required:")
            click.echo(f"   {info['auth_instructions']}")
        
        if info.get("capabilities"):
            caps = info["capabilities"]
            if caps.get("tools"):
                click.echo(f"🛠️  Available Tools: {', '.join(caps['tools'])}")
            if caps.get("resources"):
                click.echo(f"📂 Available Resources: {len(caps['resources'])}")
            if caps.get("prompts"):
                click.echo(f"📝 Available Prompts: {', '.join(caps['prompts'])}")
        
        if info.get("last_error"):
            click.echo(f"❌ Last Error: {info['last_error']}")
    
    except Exception as e:
        click.echo(f"❌ Error getting server info: {e}", err=True)
        sys.exit(1)

@mcp.command('enable')
@click.argument('server_name')
@click.option('--github-token', help='GitHub personal access token')
@click.option('--slack-bot-token', help='Slack bot token')
@click.option('--slack-app-token', help='Slack app token')
@click.option('--aws-region', help='AWS region')
@click.option('--postgres-url', help='PostgreSQL connection URL')
@click.option('--sqlite-path', help='SQLite database file path')
@click.option('--filesystem-paths', help='Comma-separated allowed filesystem paths')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def enable_mcp_server(server_name, github_token, slack_bot_token, slack_app_token, 
                     aws_region, postgres_url, sqlite_path, filesystem_paths, verbose):
    """Enable and configure an MCP server"""
    try:
        manager = get_mcp_manager(verbose=verbose)
        
        # Prepare configuration based on server type
        config = {}
        env = {}
        args = []
        
        if server_name == "github" and github_token:
            env["GITHUB_PERSONAL_ACCESS_TOKEN"] = github_token
        elif server_name == "slack":
            if slack_bot_token:
                env["SLACK_BOT_TOKEN"] = slack_bot_token
            if slack_app_token:
                env["SLACK_APP_TOKEN"] = slack_app_token
        elif server_name == "aws" and aws_region:
            env["AWS_DEFAULT_REGION"] = aws_region
        elif server_name == "postgres" and postgres_url:
            args.append(postgres_url)
        elif server_name == "sqlite" and sqlite_path:
            args.append(sqlite_path)
        elif server_name == "filesystem" and filesystem_paths:
            args.extend(filesystem_paths.split(','))
        
        if env:
            config["env"] = env
        if args:
            config["args"] = args
        
        # Enable the server
        manager.enable_server(server_name, **config)
        
        # Set up authentication if needed
        auth_params = {}
        if env:
            auth_params["env"] = env
        if args:
            auth_params["args"] = args
        
        if auth_params:
            manager.setup_authentication(server_name, **auth_params)
        
        click.echo(f"✅ Enabled MCP server: {server_name}")
        
        # Show next steps
        server_info = manager.get_server_info(server_name)
        if server_info and server_info.get("requires_auth") and not auth_params:
            click.echo("🔐 Authentication required:")
            click.echo(f"   {server_info['auth_instructions']}")
            click.echo("   Use 'ideaweaver mcp setup-auth' to configure authentication.")
    
    except Exception as e:
        click.echo(f"❌ Error enabling server: {e}", err=True)
        sys.exit(1)

@mcp.command('disable')
@click.argument('server_name')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def disable_mcp_server(server_name, verbose):
    """Disable an MCP server"""
    try:
        manager = get_mcp_manager(verbose=verbose)
        manager.disable_server(server_name)
        click.echo(f"❌ Disabled MCP server: {server_name}")
    
    except Exception as e:
        click.echo(f"❌ Error disabling server: {e}", err=True)
        sys.exit(1)

@mcp.command('setup-auth')
@click.argument('server_name')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def setup_mcp_auth(server_name, verbose):
    """Set up authentication for an MCP server"""
    try:
        manager = get_mcp_manager(verbose=verbose)
        
        # Get server info
        server_info = manager.get_server_info(server_name)
        if not server_info:
            click.echo(f"❌ Server '{server_name}' not found.", err=True)
            sys.exit(1)
        
        if not server_info.get("requires_auth"):
            click.echo(f"✅ Server '{server_name}' does not require authentication.")
            return
        
        click.echo(f"🔐 Setting up authentication for {server_info['display_name']}")
        click.echo(f"Instructions: {server_info['auth_instructions']}")
        click.echo()
        
        # Interactive setup based on server type
        auth_params = {}
        
        if server_name == "github":
            token = click.prompt("GitHub Personal Access Token", hide_input=True)
            auth_params["env"] = {"GITHUB_PERSONAL_ACCESS_TOKEN": token}
        
        elif server_name == "slack":
            bot_token = click.prompt("Slack Bot Token", hide_input=True)
            app_token = click.prompt("Slack App Token", hide_input=True)
            auth_params["env"] = {
                "SLACK_BOT_TOKEN": bot_token,
                "SLACK_APP_TOKEN": app_token
            }
        
        elif server_name == "postgres":
            postgres_url = click.prompt("PostgreSQL Connection URL")
            auth_params["args"] = [postgres_url]
        
        elif server_name == "sqlite":
            sqlite_path = click.prompt("SQLite Database Path")
            auth_params["args"] = [sqlite_path]
        
        elif server_name == "filesystem":
            paths = click.prompt("Allowed Filesystem Paths (comma-separated)")
            auth_params["args"] = paths.split(',')
        
        # Set up authentication
        if auth_params:
            success = manager.setup_authentication(server_name, **auth_params)
            if success:
                click.echo("✅ Authentication configured successfully!")
            else:
                click.echo("❌ Failed to configure authentication.", err=True)
                sys.exit(1)
    
    except Exception as e:
        click.echo(f"❌ Error setting up authentication: {e}", err=True)
        sys.exit(1)

@mcp.command('test-connection')
@click.argument('server_name')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def test_mcp_connection(server_name, verbose):
    """Test connection to an MCP server"""
    async def _test_connection():
        try:
            manager = get_mcp_manager(verbose=verbose)
            
            click.echo(f"🔗 Testing connection to {server_name}...")
            success = await manager.connect_server(server_name)
            
            if success:
                click.echo(f"✅ Successfully connected to {server_name}")
                
                # Show capabilities
                connection = manager.connections.get(server_name)
                if connection and connection.capabilities:
                    caps = connection.capabilities
                    if caps.get("tools"):
                        click.echo(f"🛠️  Available tools: {', '.join(caps['tools'])}")
                    if caps.get("resources"):
                        click.echo(f"📂 Available resources: {len(caps['resources'])}")
                    if caps.get("prompts"):
                        click.echo(f"📝 Available prompts: {', '.join(caps['prompts'])}")
                
                # Disconnect
                manager.disconnect_server(server_name)
                return True
            else:
                click.echo(f"❌ Failed to connect to {server_name}")
                return False
        
        except Exception as e:
            click.echo(f"❌ Error testing connection: {e}", err=True)
            if verbose:
                import traceback
                traceback.print_exc()
            return False
    
    # Run async function and handle result
    try:
        success = asyncio.run(_test_connection())
        if not success:
            sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)

@mcp.command('call-tool')
@click.argument('server_name')
@click.argument('tool_name')
@click.option('--args', '-a', help='JSON string of tool arguments')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--raw', is_flag=True, help='Show raw result')
def call_mcp_tool(server_name, tool_name, args, verbose, raw):
    """Call a tool on an MCP server"""
    async def _call_tool():
        try:
            # Parse arguments
            arguments = {}
            if args:
                try:
                    arguments = json.loads(args)
                except json.JSONDecodeError as e:
                    click.echo(f"❌ Invalid JSON arguments: {e}", err=True)
                    return False
            
            click.echo(f"🛠️  Calling {tool_name} on {server_name}...")
            if verbose:
                click.echo(f"Arguments: {arguments}")
            
            result = await execute_mcp_tool(server_name, tool_name, arguments, verbose=verbose)
            
            # Display result
            output = format_mcp_result(result, show_raw=raw)
            click.echo(output)
            
            return result.get("success", False)
        
        except Exception as e:
            click.echo(f"❌ Error calling tool: {e}", err=True)
            if verbose:
                import traceback
                traceback.print_exc()
            return False
    
    # Run async function and handle result
    try:
        success = asyncio.run(_call_tool())
        if not success:
            sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)

@mcp.command('status')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def show_mcp_status(verbose):
    """Show status of all MCP connections"""
    try:
        manager = get_mcp_manager(verbose=verbose)
        status = manager.get_connection_status()
        
        if not status:
            click.echo("No active MCP connections.")
            return
        
        click.echo(f"📡 MCP Connection Status ({len(status)} connections):\n")
        
        for server_name, info in status.items():
            status_icon = "🟢" if info["status"] == "connected" else "🔴"
            click.echo(f"{status_icon} {server_name}: {info['status']}")
            
            if info.get("capabilities"):
                caps = info["capabilities"]
                if caps.get("tools"):
                    click.echo(f"   🛠️  Tools: {', '.join(caps['tools'])}")
                if caps.get("resources"):
                    click.echo(f"   📂 Resources: {len(caps['resources'])}")
            
            if info.get("last_error"):
                click.echo(f"   ❌ Error: {info['last_error']}")
            
            click.echo()
    
    except Exception as e:
        click.echo(f"❌ Error getting status: {e}", err=True)
        sys.exit(1)

@mcp.command('check-deps')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def check_mcp_dependencies(verbose):
    """Check MCP dependencies and requirements"""
    try:
        manager = get_mcp_manager(verbose=verbose)
        deps = manager.check_dependencies()
        
        click.echo("🔍 MCP Dependencies Check:\n")
        
        all_good = True
        for dep, available in deps.items():
            icon = "✅" if available else "❌"
            click.echo(f"{icon} {dep}: {'Available' if available else 'Missing'}")
            if not available:
                all_good = False
        
        click.echo()
        
        if not all_good:
            click.echo("❌ Some dependencies are missing. Install them:")
            if not deps.get("mcp", False):
                click.echo("   pip install mcp")
            if not deps.get("node", False):
                click.echo("   Install Node.js: https://nodejs.org/")
            if not deps.get("npm", False):
                click.echo("   npm is usually included with Node.js")
            if not deps.get("npx", False):
                click.echo("   npx is usually included with npm")
        else:
            click.echo("✅ All dependencies are available!")
    
    except Exception as e:
        click.echo(f"❌ Error checking dependencies: {e}", err=True)
        sys.exit(1)

@mcp.command('generate-claude-config')
@click.option('--output', '-o', help='Output file path (default: claude_desktop_config.json)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def generate_claude_config(output, verbose):
    """Generate Claude Desktop configuration for enabled MCP servers"""
    try:
        manager = get_mcp_manager(verbose=verbose)
        
        if not output:
            output = "claude_desktop_config.json"
        
        config = manager.generate_claude_desktop_config(output)
        
        if config.get("mcpServers"):
            click.echo(f"📝 Generated Claude Desktop configuration with {len(config['mcpServers'])} servers:")
            for server_name in config["mcpServers"]:
                click.echo(f"   - {server_name}")
            click.echo(f"\nConfiguration saved to: {output}")
            click.echo("\n💡 To use with Claude Desktop:")
            click.echo("   1. Copy the configuration to your Claude Desktop config")
            click.echo("   2. Restart Claude Desktop")
            click.echo("   3. Servers will be available in Claude conversations")
        else:
            click.echo("ℹ️  No enabled MCP servers found. Use 'ideaweaver mcp enable <server>' first.")
    
    except Exception as e:
        click.echo(f"❌ Error generating configuration: {e}", err=True)
        sys.exit(1)

@mcp.command('demo')
@click.option('--server', default='time', help='Server to demo (default: time)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def demo_mcp(server, verbose):
    """Run a demo of MCP server functionality"""
    import asyncio
    
    async def _demo():
        try:
            manager = get_mcp_manager(verbose=verbose)
            
            if server == 'time':
                click.echo("🕐 Time Server Demo")
                click.echo("Enabling and testing time server...")
                
                # Enable server
                manager.enable_server('time')
                
                # Test connection
                success = await manager.connect_server('time')
                if success:
                    click.echo("✅ Connected successfully!")
                    
                    # Call get_current_time
                    result = await manager.call_tool('time', 'get_current_time', {})
                    if result.get('success'):
                        click.echo(f"⏰ Current time: {result.get('result')}")
                    else:
                        click.echo(f"❌ Failed: {result.get('error')}")
                else:
                    click.echo("❌ Failed to connect")
            
            elif server == 'memory':
                click.echo("🧠 Memory Server Demo")
                click.echo("Enabling and testing memory server...")
                
                manager.enable_server('memory')
                success = await manager.connect_server('memory')
                if success:
                    click.echo("✅ Connected successfully!")
                    
                    # Create a memory entity
                    result = await manager.call_tool('memory', 'create_entities', {
                        'entities': [{'name': 'demo_memory', 'entityType': 'concept', 'observations': ['This is a demo memory']}]
                    })
                    if result.get('success'):
                        click.echo("💾 Created demo memory entity")
                    else:
                        click.echo(f"❌ Failed: {result.get('error')}")
                else:
                    click.echo("❌ Failed to connect")
            
            elif server == 'terraform':
                click.echo("🏗️ Terraform Server Demo")
                click.echo("Enabling and testing Terraform server...")
                
                manager.enable_server('terraform')
                success = await manager.connect_server('terraform')
                if success:
                    click.echo("✅ Connected successfully!")
                    
                    # Search for VPC modules
                    result = await manager.call_tool('terraform', 'searchModules', {
                        'moduleQuery': 'vpc'
                    })
                    if result.get('success'):
                        click.echo("📦 Found Terraform VPC modules:")
                        formatted_result = format_mcp_result(result)
                        click.echo(formatted_result[:500] + "..." if len(formatted_result) > 500 else formatted_result)
                    else:
                        click.echo(f"❌ Failed: {result.get('error')}")
                else:
                    click.echo("❌ Failed to connect")
            
            else:
                click.echo(f"❌ Unknown server: {server}")
                click.echo("Available demo servers: time, memory, terraform")
        
        except Exception as e:
            click.echo(f"❌ Demo failed: {e}", err=True)
    
    asyncio.run(_demo())

@mcp.command('terraform-help')
def terraform_help():
    """Show comprehensive Terraform MCP server help and examples"""
    help_text = """
🏗️ **Terraform MCP Server - Complete Usage Guide**

The Terraform MCP Server provides access to the Terraform Registry for finding 
providers, modules, and documentation.

═══════════════════════════════════════════════════════════════════════════════

📋 **Available Tools:**

1. 📦 searchModules        - Search for Terraform modules
2. 📋 moduleDetails        - Get detailed module information  
3. 📚 resolveProviderDocID - Find provider documentation IDs
4. 📖 getProviderDocs      - Get specific provider documentation

═══════════════════════════════════════════════════════════════════════════════

🚀 **Setup:**

# Enable the Terraform server (uses Docker)
ideaweaver mcp enable terraform

# Check if Docker is running
docker --version

═══════════════════════════════════════════════════════════════════════════════

📦 **1. MODULE SEARCH EXAMPLES:**

# Search for VPC modules
ideaweaver mcp call-tool terraform searchModules --args '{"moduleQuery":"vpc"}'

# Search for Kubernetes modules  
ideaweaver mcp call-tool terraform searchModules --args '{"moduleQuery":"kubernetes"}'

# Search for RDS database modules
ideaweaver mcp call-tool terraform searchModules --args '{"moduleQuery":"rds"}'

# Search for Lambda function modules
ideaweaver mcp call-tool terraform searchModules --args '{"moduleQuery":"lambda"}'

# Search for security group modules
ideaweaver mcp call-tool terraform searchModules --args '{"moduleQuery":"security-group"}'

═══════════════════════════════════════════════════════════════════════════════

📋 **2. MODULE DETAILS EXAMPLES:**

# Get details for AWS VPC module
ideaweaver mcp call-tool terraform moduleDetails \\
  --args '{"moduleID":"terraform-aws-modules/vpc/aws/5.21.0"}'

# Get details for Google Cloud VPC module  
ideaweaver mcp call-tool terraform moduleDetails \\
  --args '{"moduleID":"terraform-google-modules/network/google/11.1.1"}'

# Get details for CloudPosse VPC module
ideaweaver mcp call-tool terraform moduleDetails \\
  --args '{"moduleID":"cloudposse/vpc/aws/2.2.0"}'

═══════════════════════════════════════════════════════════════════════════════

📚 **3. PROVIDER DOCUMENTATION EXAMPLES:**

# Find AWS provider documentation
ideaweaver mcp call-tool terraform resolveProviderDocID \\
  --args '{"serviceSlug":"aws"}'

# Find Azure provider documentation
ideaweaver mcp call-tool terraform resolveProviderDocID \\
  --args '{"serviceSlug":"azurerm"}'

# Find Google Cloud provider documentation  
ideaweaver mcp call-tool terraform resolveProviderDocID \\
  --args '{"serviceSlug":"google"}'

# Find random provider documentation (simple provider)
ideaweaver mcp call-tool terraform resolveProviderDocID \\
  --args '{"serviceSlug":"random"}'

# Find local provider documentation
ideaweaver mcp call-tool terraform resolveProviderDocID \\
  --args '{"serviceSlug":"local"}'

═══════════════════════════════════════════════════════════════════════════════

📖 **4. GET PROVIDER DOCS EXAMPLES:**

# First get a document ID using resolveProviderDocID, then:
ideaweaver mcp call-tool terraform getProviderDocs \\
  --args '{"docId":"DOCUMENT_ID_FROM_PREVIOUS_COMMAND"}'

═══════════════════════════════════════════════════════════════════════════════

🎯 **COMMON WORKFLOWS:**

**Workflow 1: Find and explore a module**
1. ideaweaver mcp call-tool terraform searchModules --args '{"moduleQuery":"s3"}'
2. Pick a moduleID from results
3. ideaweaver mcp call-tool terraform moduleDetails --args '{"moduleID":"selected-module-id"}'

**Workflow 2: Explore provider capabilities**  
1. ideaweaver mcp call-tool terraform resolveProviderDocID --args '{"serviceSlug":"aws"}'
2. Pick a docId from results
3. ideaweaver mcp call-tool terraform getProviderDocs --args '{"docId":"selected-doc-id"}'

═══════════════════════════════════════════════════════════════════════════════

⚡ **QUICK TEST COMMANDS:**

# Test connection
ideaweaver mcp test-connection terraform

# Quick module search
ideaweaver mcp call-tool terraform searchModules --args '{"moduleQuery":"hello"}'

# Get popular AWS VPC module details
ideaweaver mcp call-tool terraform moduleDetails \\
  --args '{"moduleID":"terraform-aws-modules/vpc/aws/5.21.0"}'

═══════════════════════════════════════════════════════════════════════════════

🔧 **TROUBLESHOOTING:**

If commands fail:
1. Check Docker is running: docker --version
2. Enable server: ideaweaver mcp enable terraform  
3. Test connection: ideaweaver mcp test-connection terraform
4. Check status: ideaweaver mcp status

═══════════════════════════════════════════════════════════════════════════════

📊 **POPULAR MODULE QUERIES:**
- "vpc"           - Virtual Private Cloud modules
- "eks"           - Kubernetes cluster modules  
- "rds"           - Database modules
- "lambda"        - Serverless function modules
- "alb"           - Load balancer modules
- "s3"            - Storage bucket modules
- "iam"           - Identity and access modules
- "security"      - Security group modules
- "monitoring"    - CloudWatch/monitoring modules
- "backup"        - Backup solution modules

═══════════════════════════════════════════════════════════════════════════════

🌟 **PRO TIPS:**
- Module search is case-insensitive
- Use specific terms for better results ("eks" vs "kubernetes")
- moduleDetails shows inputs, outputs, and examples
- Provider docs give you resource capabilities
- All output is formatted for easy reading

═══════════════════════════════════════════════════════════════════════════════

💡 For more help: ideaweaver mcp --help
🔗 Terraform Registry: https://registry.terraform.io/
📚 HashiCorp Docs: https://github.com/hashicorp/terraform-mcp-server

    """
    click.echo(help_text)

@click.group()
def agent():
    """Intelligent agent workflows for creative and analytical tasks.
    
    This command group provides access to various AI agents that can help with:
    - Creative writing and content generation
    - Research and analysis
    - Social media content creation
    - Travel planning and recommendations
    - Financial analysis and insights
    
    Each agent is specialized for its specific task and can work independently
    or collaborate with other agents for complex workflows.
    """
    pass

@agent.command('generate_storybook')
@click.option('--theme', required=True, help='Theme of the storybook')
@click.option('--target-age', required=True, help='Target age group (e.g., "3-5", "6-8", "9-12")')
@click.option('--num-pages', default=1, help='Number of pages in the storybook')
@click.option('--style', default='whimsical', help='Writing style (e.g., "whimsical", "educational", "adventure")')
@click.option('--openai-api-key', envvar='OPENAI_API_KEY', help='OpenAI API key (fallback if Ollama not available)')
@click.option('--force-openai', is_flag=True, help='Force use of OpenAI instead of Ollama')
def generate_storybook(theme, target_age, num_pages, style, openai_api_key, force_openai):
    """Generate an engaging and age-appropriate storybook.
    
    This agent creates complete storybooks with:
    - Age-appropriate content and vocabulary
    - Engaging narrative structure
    - Educational elements
    - Character development
    - Moral lessons or themes
    
    The agent considers the target age group to ensure appropriate:
    - Language complexity
    - Story length
    - Themes and concepts
    - Educational value
    """
    import subprocess
    import sys
    from pathlib import Path
    
    click.echo(f"🚀 Generating storybook: '{theme}' for ages {target_age}")
    click.echo("📋 LLM Priority: Ollama (local) → OpenAI (cloud)")
    
    # Prepare the command to run in the current environment
    cmd = [
        'python', '-c',
        f"from ideaweaver.crew_ai import StorybookGenerator; "
        f"generator = StorybookGenerator(openai_api_key={repr(openai_api_key)}); "
        f"result = generator.create_storybook("
        f"theme='{theme}', target_age='{target_age}', "
        f"num_pages={num_pages}, style='{style}'); "
        f"print(result.get('formatted_content', result.get('content', 'No content generated')))"
    ]
    
    try:
        import os
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path.cwd())
        if openai_api_key:
            env["OPENAI_API_KEY"] = openai_api_key
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        click.echo(result.stdout)
        if result.stderr:
            click.echo(f"Warnings: {result.stderr}", err=True)
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Error running storybook generation: {e.stderr}", err=True)
        if "No LLM available" in str(e.stderr):
            click.echo("\n💡 Setup Instructions:")
            click.echo("   Option 1 (Recommended): Install Ollama")
            click.echo("     1. Download from: https://ollama.com/")
            click.echo("     2. Install a model: ollama pull llama3.2:3b")
            click.echo("     3. Start Ollama: ollama serve")
            click.echo("   Option 2: Use OpenAI")
            click.echo("     Set OPENAI_API_KEY environment variable")
        sys.exit(1)

@agent.command('research_write')
@click.option('--topic', required=True, help='Research topic to investigate and write about')
@click.option('--content-type', default='blog post', help='Type of content (blog post, article, report)')
@click.option('--audience', default='tech enthusiasts', help='Target audience for the content')
@click.option('--openai-api-key', envvar='OPENAI_API_KEY', help='OpenAI API key')
def research_write(topic, content_type, audience, openai_api_key):
    """Research and write comprehensive content on any topic.
    
    This agent performs:
    - In-depth topic research
    - Information synthesis
    - Content structuring
    - Audience-appropriate writing
    - Fact verification
    
    The agent adapts its writing style and depth based on:
    - Content type (blog, article, report)
    - Target audience expertise
    - Topic complexity
    - Required format
    """
    import subprocess
    import sys
    from pathlib import Path
    
    click.echo(f"🔍 Researching and writing about: '{topic}'")
    click.echo(f"📝 Content Type: {content_type} | 👥 Audience: {audience}")
    click.echo("📋 LLM Priority: Ollama (local) → OpenAI (cloud)")
    
    # Prepare the command to run in the current environment
    cmd = [
        'python', '-c',
        f"from ideaweaver.crew_ai import ResearchWriterGenerator; "
        f"generator = ResearchWriterGenerator(openai_api_key={repr(openai_api_key)}); "
        f"result = generator.create_research_content("
        f"topic='{topic}', content_type='{content_type}', "
        f"target_audience='{audience}'); "
        f"print(result.get('formatted_content', result.get('content', 'No content generated')))"
    ]
    
    try:
        import os
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path.cwd())
        if openai_api_key:
            env["OPENAI_API_KEY"] = openai_api_key
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        click.echo("\n" + "="*80)
        click.echo(f"📄 RESEARCH & WRITING: {topic}")
        click.echo(f"📝 Type: {content_type} | 👥 Audience: {audience}")
        click.echo("="*80)
        click.echo(result.stdout)
        click.echo("="*80)
        if result.stderr:
            click.echo(f"Warnings: {result.stderr}", err=True)
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Error running research and writing: {e.stderr}", err=True)
        sys.exit(1)

@agent.command('linkedin_post')
@click.option('--topic', required=True, help='Topic for the LinkedIn post')
@click.option('--post-type', default='professional insights', help='Type of post (professional insights, career advice, industry trends)')
@click.option('--tone', default='engaging', help='Tone of the post (engaging, professional, inspirational, thought-provoking)')
@click.option('--openai-api-key', envvar='OPENAI_API_KEY', help='OpenAI API key')
def linkedin_post(topic, post_type, tone, openai_api_key):
    """Create engaging and professional LinkedIn content.
    
    This agent crafts LinkedIn posts that:
    - Drive professional engagement
    - Share valuable insights
    - Build thought leadership
    - Encourage meaningful discussions
    
    The agent optimizes content for:
    - Professional audience
    - Platform best practices
    - Engagement metrics
    - Brand voice consistency
    """
    import subprocess
    import sys
    from pathlib import Path
    
    click.echo(f"📱 Creating LinkedIn post about: '{topic}'")
    click.echo(f"📝 Type: {post_type} | 🎯 Tone: {tone}")
    click.echo("📋 LLM Priority: Ollama (local) → OpenAI (cloud)")
    
    # Prepare the command to run in the current environment
    cmd = [
        'python', '-c',
        f"from ideaweaver.crew_ai import LinkedInPostGenerator; "
        f"generator = LinkedInPostGenerator(openai_api_key={repr(openai_api_key)}); "
        f"result = generator.create_linkedin_post("
        f"topic='{topic}', post_type='{post_type}', "
        f"tone='{tone}'); "
        f"print(result.get('formatted_content', result.get('content', 'No content generated')))"
    ]
    
    try:
        import os
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path.cwd())
        if openai_api_key:
            env["OPENAI_API_KEY"] = openai_api_key
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        click.echo("\n" + "="*80)
        click.echo(f"📱 LINKEDIN POST: {topic}")
        click.echo(f"📝 Type: {post_type} | 🎯 Tone: {tone}")
        click.echo("="*80)
        click.echo(result.stdout)
        click.echo("="*80)
        if result.stderr:
            click.echo(f"Warnings: {result.stderr}", err=True)
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Error running LinkedIn post generation: {e.stderr}", err=True)
        sys.exit(1)

@agent.command('stock_analysis')
@click.option('--symbol', required=True, help='Stock ticker symbol (e.g., "AAPL")')
@click.option('--openai-api-key', envvar='OPENAI_API_KEY', help='OpenAI API key')
def stock_analysis(symbol, openai_api_key):
    """Analyze stock performance, trends, and provide investment insights."""
    import subprocess
    import sys
    from pathlib import Path
    
    click.echo(f"📈 Analyzing stock: {symbol}")
    click.echo("📋 LLM Priority: Ollama (local) → OpenAI (cloud)")
    
    # Prepare the command to run in the current environment
    cmd = [
        'python', '-c',
        f"from ideaweaver.crew_ai import StockAnalysisGenerator; "
        f"generator = StockAnalysisGenerator(openai_api_key={repr(openai_api_key)}); "
        f"result = generator.analyze_stock('{symbol}'); "
        f"print(result)"
    ]
    
    try:
        import os
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path.cwd())
        if openai_api_key:
            env["OPENAI_API_KEY"] = openai_api_key
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        click.echo("\n📊 Stock Analysis Results:")
        click.echo("=" * 80)
        click.echo(result.stdout)
        click.echo("=" * 80)
        if result.stderr:
            click.echo(f"Warnings: {result.stderr}", err=True)
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Error analyzing stock: {e.stderr}", err=True)
        sys.exit(1)

@agent.command('travel_plan')
@click.option('--destination', required=True, help='Travel destination')
@click.option('--duration', required=True, help='Trip duration (e.g., "5 days", "2 weeks")')
@click.option('--budget', required=True, help='Budget range (e.g., "$1000-2000", "luxury")')
@click.option('--preferences', default='balanced', help='Travel style preferences (e.g., "adventure", "relaxed", "balanced")')
@click.option('--openai-api-key', envvar='OPENAI_API_KEY', help='OpenAI API key')
def travel_plan(destination, duration, budget, preferences, openai_api_key):
    """Create personalized travel itineraries with recommendations for activities, accommodations, and local experiences."""
    import subprocess
    import sys
    from pathlib import Path
    
    click.echo(f"✈️ Creating travel plan for: {destination}")
    click.echo(f"⏱️ Duration: {duration} | 💰 Budget: {budget} | 🎯 Style: {preferences}")
    click.echo("📋 LLM Priority: Ollama (local) → OpenAI (cloud)")
    
    # Prepare the command to run in the current environment
    api_key_arg = f", openai_api_key='{openai_api_key}'" if openai_api_key else ""
    cmd = [
        'python', '-c',
        f"from ideaweaver.crew_ai import TravelPlannerGenerator; "
        f"generator = TravelPlannerGenerator(openai_api_key={repr(openai_api_key)}); "
        f"result = generator.create_travel_plan("
        f"destination='{destination}', duration='{duration}', "
        f"budget='{budget}', preferences='{preferences}'); "
        f"print(result)"
    ]
    
    try:
        import os
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path.cwd())
        if openai_api_key:
            env["OPENAI_API_KEY"] = openai_api_key
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        click.echo("\n📋 Your Travel Plan:")
        click.echo("=" * 80)
        click.echo(result.stdout)
        click.echo("=" * 80)
        if result.stderr:
            click.echo(f"Warnings: {result.stderr}", err=True)
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Error creating travel plan: {e.stderr}", err=True)
        sys.exit(1)

@agent.command('check-llm')
def check_llm_status():
    """Check the status and availability of language model providers.
    
    This command verifies:
    - Local model availability (Ollama)
    - API access (OpenAI)
    - Model capabilities
    - Connection status
    
    Useful for troubleshooting and ensuring required models are available
    for agent operations.
    """
    from .crew_ai import setup_intelligent_llm
    
    try:
        llm, llm_type, model_used = setup_intelligent_llm()
        click.echo(f"\n✅ LLM Status:")
        click.echo(f"   Provider: {llm_type}")
        click.echo(f"   Model: {model_used}")
        click.echo(f"   Status: Connected and ready")
    except Exception as e:
        click.echo(f"❌ Error checking LLM status: {e}", err=True)
        sys.exit(1)

# Add the agent group to the main CLI
cli.add_command(agent)

# @cli.group()
# def bedrock():
#     """AWS Bedrock integration commands"""
#     pass

# @bedrock.command('setup-guide')
# def bedrock_setup_guide():
#     """Show AWS Bedrock setup instructions"""
#     try:
#         from .trainer import ModelTrainer
#         trainer = ModelTrainer({}, verbose=False)
#         trainer.show_bedrock_setup_guide()
#     except Exception as e:
#         click.echo(f"Error displaying setup guide: {e}", err=True)

# @bedrock.command('validate-model')
# @click.argument('model_path')
# @click.option('--verbose', '-v', is_flag=True, help='Verbose output')
# def validate_bedrock_model(model_path, verbose):
#     """Validate if a model is compatible with AWS Bedrock"""
#     try:
#         from .aws_bedrock import BedrockModelImporter
        
#         click.echo(f"🔍 Validating model for Bedrock compatibility: {model_path}")
        
#         importer = BedrockModelImporter()
        
#         # Check architecture
#         arch_valid = importer.validate_model_architecture(model_path)
#         if arch_valid:
#             click.echo("✅ Model architecture appears compatible")
#         else:
#             click.echo("❌ Model architecture may not be supported")
        
#         # Check files
#         file_validation = importer.validate_model_files(model_path)
        
#         click.echo("\n📁 File validation results:")
#         for file_type, exists in file_validation.items():
#             status = "✅" if exists else "❌"
#             click.echo(f"   {status} {file_type}")
        
#         required_files = ['safetensors', 'config', 'tokenizer_config', 'tokenizer_json']
#         missing_files = [f for f in required_files if not file_validation[f]]
        
#         if missing_files:
#             click.echo(f"\n❌ Missing required files: {missing_files}")
#             click.echo("   Models must have .safetensors, config.json, tokenizer_config.json, tokenizer.json")
#         else:
#             click.echo("\n✅ All required files present!")
            
#         # Check model size
#         try:
#             import os
#             from pathlib import Path
            
#             model_dir = Path(model_path)
#             total_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
#             size_gb = total_size / (1024**3)
            
#             click.echo(f"\n📏 Model size: {size_gb:.2f} GB")
            
#             if size_gb > 200:
#                 click.echo("❌ Model too large for Bedrock (max 200GB for text models)")
#             elif size_gb > 100:
#                 click.echo("⚠️  Model may be too large for multimodal support (max 100GB)")
#             else:
#                 click.echo("✅ Model size is within Bedrock limits")
                
#         except Exception as e:
#             if verbose:
#                 click.echo(f"⚠️  Could not calculate model size: {e}")
        
#     except ImportError:
#         click.echo("❌ AWS Bedrock integration not available. Install with: pip install boto3", err=True)
#     except Exception as e:
#         click.echo(f"❌ Validation error: {e}", err=True)
#         if verbose:
#             import traceback
#             traceback.print_exc()

# @bedrock.command('test-import')
# @click.argument('model_path')
# @click.option('--model-name', required=True, help='Bedrock model name')
# @click.option('--s3-bucket', required=True, help='S3 bucket for model storage')
# @click.option('--role-arn', required=True, help='IAM role ARN for Bedrock import')
# @click.option('--region', default='us-east-1', help='AWS region')
# @click.option('--s3-prefix', help='S3 prefix/folder for model files')
# @click.option('--job-name', help='Custom import job name')
# @click.option('--test-inference', is_flag=True, default=True, help='Test model inference after import')
# @click.option('--verbose', '-v', is_flag=True, help='Verbose output')
# def test_bedrock_import(model_path, model_name, s3_bucket, role_arn, region, 
#                        s3_prefix, job_name, test_inference, verbose):
#     """Test importing a model to AWS Bedrock"""
#     try:
#         from .aws_bedrock import BedrockModelImporter
        
#         click.echo(f"🚀 Testing Bedrock import for model: {model_path}")
        
#         importer = BedrockModelImporter(region_name=region)
        
#         results = importer.import_model_to_bedrock(
#             model_path=model_path,
#             model_name=model_name,
#             s3_bucket=s3_bucket,
#             role_arn=role_arn,
#             s3_prefix=s3_prefix,
#             job_name=job_name,
#             test_inference=test_inference
#         )
        
#         if results['status'] == 'COMPLETED':
#             click.echo("✅ Bedrock import test completed successfully!")
#             click.echo(f"🎯 Model ID: {results['model_id']}")
            
#             if results.get('test_response'):
#                 click.echo(f"🧪 Test response: {results['test_response']}")
#         else:
#             click.echo(f"❌ Bedrock import test failed: {results.get('error', 'Unknown error')}")
            
#     except ImportError:
#         click.echo("❌ AWS Bedrock integration not available. Install with: pip install boto3", err=True)
#     except Exception as e:
#         click.echo(f"❌ Import test error: {e}", err=True)
#         if verbose:
#             import traceback
#             traceback.print_exc()

#cli.add_command(bedrock)

if __name__ == '__main__':
    cli() 