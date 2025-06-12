"""
IdeaWeaver Training Module
Provides comprehensive model training with AutoTrain Advanced integration
"""

import os
import subprocess
import tempfile
import yaml
import shutil
import re
import json
import glob
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
from .config import create_ideaweaver_config, load_config

# Import Bedrock integration
try:
    from .aws_bedrock import BedrockModelImporter, create_bedrock_iam_role_guide
    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False
    BedrockModelImporter = None

class ModelTrainer:
    """Model training using with Hugging Face Hub integration"""
    
    def __init__(self, config: Dict[str, Any], verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self.ideaweaver_config = create_ideaweaver_config(config)
    
    def train(self) -> Optional[str]:
        """Start model training"""
        
        # Setup experiment tracking
        comet_experiment, mlflow_run = self._setup_experiment_tracking()
        
        try:
            # Create project directory for training
            project_dir = f"./autotrain_projects/{self.config['project_name']}"
            os.makedirs(project_dir, exist_ok=True)
            
            # Copy dataset to project directory with standard name
            original_dataset = self.config['dataset']
            train_csv_path = os.path.join(project_dir, 'train.csv')
            shutil.copy2(original_dataset, train_csv_path)
            
            # Update config to point to project directory
            self.config['dataset'] = project_dir
            
            # Convert to training format
            self.ideaweaver_config = create_ideaweaver_config(self.config)
            
            # Create temporary training config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
                yaml.dump(self.ideaweaver_config, f, default_flow_style=False)
                temp_config_path = f.name
            
            if self.verbose:
                print(f"📝 Training config written to: {temp_config_path}")
                print("📋 Training configuration:")
                print(yaml.dump(self.ideaweaver_config, default_flow_style=False))
            
            # Prepare training command
            cmd = [
                'autotrain',
                '--config', temp_config_path
            ]
            
            if self.verbose:
                print(f"🔧 Running command: {' '.join(cmd)}")
            
            # Run training
            env = os.environ.copy()
            env['AUTOTRAIN_BACKEND'] = self.config['backend']
            
            process = subprocess.run(
                cmd,
                env=env,
                capture_output=True,  # Always capture output for summary
                text=True,
                cwd=os.getcwd()
            )
            
            # Display output if verbose
            if self.verbose and process.stdout:
                print(process.stdout)
            if self.verbose and process.stderr:
                print(process.stderr)
            
            # Clean up temp file
            os.unlink(temp_config_path)
            
            if process.returncode == 0:
                output_dir = f"./{self.config['project_name']}"
                
                # Extract metrics and log to tracking platforms
                combined_output = (process.stdout or "") + (process.stderr or "")
                metrics = self._extract_metrics_from_output(combined_output)
                
                # Log to experiment tracking platforms
                self._log_metrics_to_platforms(metrics, comet_experiment, mlflow_run)
                
                # Log the trained model to platforms
                self._log_model_to_platforms(output_dir, comet_experiment, mlflow_run)
                
                # Push to Hugging Face Hub if requested
                if self.config.get('hub', {}).get('push_to_hub', False):
                    self._push_to_hub(output_dir)
                
                # Push to AWS Bedrock if requested
                bedrock_config = self.config.get('bedrock', {})
                if bedrock_config and bedrock_config.get('model_name'):
                    try:
                        print("☁️  Deploying to AWS Bedrock...")
                        bedrock_results = self._push_to_bedrock(output_dir)
                        
                        if bedrock_results['status'] == 'COMPLETED':
                            print("✅ AWS Bedrock deployment completed successfully!")
                        else:
                            print(f"⚠️  AWS Bedrock deployment failed: {bedrock_results.get('error', 'Unknown error')}")
                            
                    except Exception as e:
                        print(f"❌ AWS Bedrock deployment error: {str(e)}")
                        if self.verbose:
                            import traceback
                            traceback.print_exc()
                
                # Display training summary
                self._display_training_summary(output_dir, combined_output)
                
                return output_dir
            else:
                if process.stderr:
                    print(f"❌ Training error: {process.stderr}")
                return None
                
        except Exception as e:
            print(f"❌ Training error: {str(e)}")
            return None
        
        finally:
            # Always finalize experiment tracking
            self._finalize_experiment_tracking(comet_experiment, mlflow_run)
    
    def _push_to_hub(self, model_path: str):
        """Push trained model to Hugging Face Hub"""
        try:
            from huggingface_hub import HfApi, login
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            hub_config = self.config.get('hub', {})
            model_id = hub_config.get('hub_model_id')
            token = hub_config.get('token') or os.environ.get('HF_TOKEN')
            
            if not model_id:
                print("❌ No hub_model_id specified for pushing to Hub")
                return False
            
            if self.verbose:
                print(f"🚀 Pushing model to Hugging Face Hub: {model_id}")
            
            # Login to Hugging Face
            if token:
                login(token=token)
            
            # Load the trained model and tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                
                # Push to hub
                tokenizer.push_to_hub(model_id, token=token)
                model.push_to_hub(model_id, token=token)
                
                print(f"✅ Model successfully pushed to: https://huggingface.co/{model_id}")
                return True
                
            except Exception as e:
                print(f"❌ Error loading model for push: {str(e)}")
                return False
                
        except ImportError:
            print("❌ huggingface_hub not installed. Install with: pip install huggingface_hub")
            return False
        except Exception as e:
            print(f"❌ Error pushing to Hub: {str(e)}")
            return False
    
    def _push_to_bedrock(self, model_path: str) -> Dict[str, Any]:
        """Push trained model to AWS Bedrock"""
        if not BEDROCK_AVAILABLE:
            raise ImportError("AWS Bedrock integration not available. Install with: pip install boto3")
        
        bedrock_config = self.config.get('bedrock', {})
        
        # Required parameters
        required_params = ['model_name', 's3_bucket', 'role_arn']
        missing_params = [p for p in required_params if not bedrock_config.get(p)]
        
        if missing_params:
            raise ValueError(f"Missing required Bedrock parameters: {missing_params}")
        
        try:
            # Initialize Bedrock importer
            region = bedrock_config.get('region', 'us-east-1')
            importer = BedrockModelImporter(region_name=region)
            
            if self.verbose:
                print(f"🔧 Initializing AWS Bedrock import...")
                print(f"   Region: {region}")
                print(f"   Model: {bedrock_config['model_name']}")
                print(f"   S3 Bucket: {bedrock_config['s3_bucket']}")
            
            # Import model to Bedrock
            results = importer.import_model_to_bedrock(
                model_path=model_path,
                model_name=bedrock_config['model_name'],
                s3_bucket=bedrock_config['s3_bucket'],
                role_arn=bedrock_config['role_arn'],
                s3_prefix=bedrock_config.get('s3_prefix'),
                job_name=bedrock_config.get('job_name'),
                test_inference=bedrock_config.get('test_inference', True)
            )
            
            if results['status'] == 'COMPLETED':
                print(f"✅ Model successfully imported to AWS Bedrock!")
                print(f"🎯 Bedrock Model ID: {results['model_id']}")
                
                if results.get('test_response'):
                    print(f"🧪 Test response: {results['test_response']}")
                
                # Save Bedrock info to the model directory
                bedrock_info = {
                    'model_id': results['model_id'],
                    'job_id': results['job_id'],
                    's3_uri': results['s3_uri'],
                    'region': region,
                    'import_timestamp': time.time()
                }
                
                bedrock_info_path = os.path.join(model_path, 'bedrock_info.json')
                with open(bedrock_info_path, 'w') as f:
                    json.dump(bedrock_info, f, indent=2)
                
                print(f"💾 Bedrock info saved to: {bedrock_info_path}")
                
            return results
            
        except Exception as e:
            error_msg = f"Failed to import model to Bedrock: {str(e)}"
            print(f"❌ {error_msg}")
            
            # Return error result
            return {
                'status': 'FAILED',
                'error': error_msg,
                'model_name': bedrock_config['model_name']
            }
    
    def show_bedrock_setup_guide(self):
        """Display Bedrock setup instructions"""
        if not BEDROCK_AVAILABLE:
            print("❌ AWS Bedrock integration not available. Install with: pip install boto3")
            return
            
        print(create_bedrock_iam_role_guide())
    
    def validate_dataset(self) -> bool:
        """Validate dataset format"""
        
        dataset_path = self.config['dataset']
        
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset not found: {dataset_path}")
            return False
        
        # Basic validation for text classification
        if self.config['task'] == 'text_classification':
            if dataset_path.endswith('.csv'):
                try:
                    import pandas as pd
                    df = pd.read_csv(dataset_path)
                    
                    # Check for required columns
                    if 'text' not in df.columns:
                        print("❌ CSV must have 'text' column")
                        return False
                    
                    if 'target' not in df.columns and 'label' not in df.columns:
                        print("❌ CSV must have 'target' or 'label' column")  
                        return False
                    
                    if self.verbose:
                        print(f"✅ Dataset validation passed ({len(df)} rows)")
                    
                    return True
                    
                except Exception as e:
                    print(f"❌ Error reading CSV: {str(e)}")
                    return False
        
        return True

    def _display_training_summary(self, output_dir: str, process_output: str = ""):
        """Display a focused training summary with key metrics"""
        try:
            print("\n" + "="*60)
            print("🎉 TRAINING SUMMARY")
            print("="*60)
            
            # Model Information
            print(f"📂 Model Path:           {output_dir}")
            print(f"🤖 Base Model:           {self.config['base_model']}")
            print(f"📊 Dataset:              {self.config['dataset']}")
            
            # Extract and display key metrics
            metrics = self._extract_metrics_from_output(process_output)
            
            print(f"\n📊 KEY PERFORMANCE METRICS")
            print("-" * 40)
            
            # Final Training Loss
            if 'final_train_loss' in metrics:
                print(f"📉 Final Train Loss:     {metrics['final_train_loss']:.4f}")
            else:
                print(f"📉 Final Train Loss:     Not available")
            
            # Overall Accuracy
            if 'eval_metrics' in metrics and 'eval_accuracy' in metrics['eval_metrics']:
                accuracy = metrics['eval_metrics']['eval_accuracy']
                print(f"🎯 Overall Accuracy:     {accuracy:.1%}")
            else:
                print(f"🎯 Overall Accuracy:     Not available")
            
            print("\n" + "="*60)
            print("✨ Training completed successfully! Model is ready for use.")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"\n📊 Training completed successfully!")
            print(f"📂 Model saved to: {output_dir}")
            if self.verbose:
                print(f"⚠️  Could not generate detailed summary: {str(e)}")
    
    def _extract_metrics_from_output(self, output: str) -> Dict[str, Any]:
        """Extract training metrics from the process output and trainer state files"""
        metrics = {}
        
        try:
            # Extract final training loss
            train_loss_match = re.search(r"'train_loss':\s*([\d.]+)", output)
            if train_loss_match:
                metrics['final_train_loss'] = float(train_loss_match.group(1))
            
            # Extract final evaluation metrics (last occurrence)
            eval_patterns = {
                'eval_accuracy': r"'eval_accuracy':\s*([\d.]+)",
                'eval_f1_macro': r"'eval_f1_macro':\s*([\d.]+)", 
                'eval_f1_micro': r"'eval_f1_micro':\s*([\d.]+)",
                'eval_precision_macro': r"'eval_precision_macro':\s*([\d.]+)",
                'eval_recall_macro': r"'eval_recall_macro':\s*([\d.]+)",
                'eval_loss': r"'eval_loss':\s*([\d.]+)"
            }
            
            eval_metrics = {}
            for metric_name, pattern in eval_patterns.items():
                matches = re.findall(pattern, output)
                if matches:
                    eval_metrics[metric_name] = float(matches[-1])  # Get last occurrence
            
            if eval_metrics:
                metrics['eval_metrics'] = eval_metrics
            
            # Extract training efficiency stats
            runtime_match = re.search(r"'train_runtime':\s*([\d.]+)", output)
            samples_per_sec_match = re.search(r"'train_samples_per_second':\s*([\d.]+)", output)
            steps_per_sec_match = re.search(r"'train_steps_per_second':\s*([\d.]+)", output)
            
            if runtime_match or samples_per_sec_match or steps_per_sec_match:
                training_stats = {}
                if runtime_match:
                    training_stats['train_runtime'] = float(runtime_match.group(1))
                if samples_per_sec_match:
                    training_stats['train_samples_per_second'] = float(samples_per_sec_match.group(1))
                if steps_per_sec_match:
                    training_stats['train_steps_per_second'] = float(steps_per_sec_match.group(1))
                metrics['training_stats'] = training_stats
        
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Error extracting metrics from output: {str(e)}")
        
        # Try to load metrics from trainer state files
        try:
            self._load_metrics_from_trainer_state(metrics)
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Error loading metrics from trainer state: {str(e)}")
        
        return metrics
    
    def _load_metrics_from_trainer_state(self, metrics: Dict[str, Any]):
        """Load metrics from trainer_state.json files in the output directory"""
        import json
        import glob
        
        # Get the project name and output directory
        project_name = self.config.get('project_name', '')
        output_dir = self.config.get('output_dir', f'./{project_name}')
        
        # Look for trainer_state.json files in various locations, prioritizing current project
        possible_paths = [
            # Current project specific paths (highest priority)
            os.path.join(output_dir, 'trainer_state.json'),
            os.path.join(output_dir, 'checkpoint-*/trainer_state.json'),
            f'./{project_name}/trainer_state.json',
            f'./{project_name}/checkpoint-*/trainer_state.json',
            # General project output directories
            './*/trainer_state.json',
            './*/*/trainer_state.json'
        ]
        
        trainer_state_files = []
        for pattern in possible_paths:
            trainer_state_files.extend(glob.glob(pattern))
        
        if not trainer_state_files:
            if self.verbose:
                print("📊 No trainer_state.json files found")
            return
        
        # Prioritize files from current project, then use most recent
        current_project_files = [f for f in trainer_state_files if project_name in f]
        
        if current_project_files:
            latest_file = max(current_project_files, key=os.path.getmtime)
            if self.verbose:
                print(f"📊 Loading metrics from current project: {latest_file}")
        else:
            latest_file = max(trainer_state_files, key=os.path.getmtime)
            if self.verbose:
                print(f"📊 Loading metrics from most recent training: {latest_file}")
        
        try:
            with open(latest_file, 'r') as f:
                trainer_state = json.load(f)
            
            # Extract final training loss from log history
            log_history = trainer_state.get('log_history', [])
            if log_history:
                # Get the last training step (not evaluation step)
                training_steps = [entry for entry in log_history if 'loss' in entry]
                if training_steps:
                    final_training_step = training_steps[-1]
                    metrics['final_train_loss'] = final_training_step.get('loss')
                
                # Get the last evaluation metrics
                eval_steps = [entry for entry in log_history if 'eval_accuracy' in entry]
                if eval_steps:
                    final_eval_step = eval_steps[-1]
                    eval_metrics = {}
                    
                    # Extract all eval metrics from the final evaluation
                    for key, value in final_eval_step.items():
                        if key.startswith('eval_'):
                            eval_metrics[key] = value
                    
                    if eval_metrics:
                        metrics['eval_metrics'] = eval_metrics
                
                # Extract additional metadata
                if 'epoch' in trainer_state:
                    metrics['final_epoch'] = trainer_state['epoch']
                if 'global_step' in trainer_state:
                    metrics['total_steps'] = trainer_state['global_step']
                if 'best_metric' in trainer_state:
                    metrics['best_metric'] = trainer_state['best_metric']
            
            if self.verbose:
                print(f"✅ Successfully loaded metrics from trainer state")
                if 'final_train_loss' in metrics:
                    print(f"   📉 Final training loss: {metrics['final_train_loss']:.4f}")
                if 'eval_metrics' in metrics and 'eval_accuracy' in metrics['eval_metrics']:
                    print(f"   🎯 Final accuracy: {metrics['eval_metrics']['eval_accuracy']:.4f}")
                    
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Error reading trainer state file {latest_file}: {str(e)}")
    
    def _get_model_file_info(self, output_dir: str) -> Dict[str, float]:
        """Get information about generated model files"""
        model_files = {}
        
        try:
            if os.path.exists(output_dir):
                important_files = [
                    'model.safetensors', 'pytorch_model.bin', 'config.json',
                    'tokenizer.json', 'vocab.txt', 'README.md'
                ]
                
                for filename in important_files:
                    filepath = os.path.join(output_dir, filename)
                    if os.path.exists(filepath):
                        size_bytes = os.path.getsize(filepath)
                        size_mb = size_bytes / (1024 * 1024)
                        model_files[filename] = size_mb
                        
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Error getting file info: {str(e)}")
        
        return model_files
    
    def _setup_experiment_tracking(self):
        """Initialize experiment tracking platforms"""
        if not self.config.get('tracking', {}).get('enabled', False):
            return None, None
        
        comet_experiment = None
        mlflow_run = None
        
        try:
            tracking_config = self.config['tracking']
            
            # Initialize DagsHub if MLflow is configured
            if tracking_config.get('dagshub', {}).get('enabled', False):
                try:
                    import dagshub
                    dagshub_config = tracking_config['dagshub']
                    dagshub.init(
                        repo_owner=dagshub_config['repo_owner'],
                        repo_name=dagshub_config['repo_name'],
                        mlflow=True
                    )
                    if self.verbose:
                        print("✅ DagsHub tracking initialized")
                except ImportError:
                    if self.verbose:
                        print("⚠️  dagshub not installed, skipping DagsHub tracking")
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  DagsHub setup failed: {str(e)}")
            
            # Initialize Comet ML
            if tracking_config.get('comet', {}).get('api_key'):
                try:
                    import comet_ml
                    comet_experiment = comet_ml.Experiment(
                        api_key=tracking_config['comet']['api_key'],
                        project_name=tracking_config['comet']['project']
                    )
                    comet_experiment.set_name(f"ideaweaver-{self.config['project_name']}")
                    if self.verbose:
                        print("✅ Comet ML tracking initialized")
                except ImportError:
                    if self.verbose:
                        print("⚠️  comet-ml not installed, skipping Comet tracking")
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Comet ML setup failed: {str(e)}")
            
            # Initialize MLflow (DagsHub compatible)
            if (tracking_config.get('mlflow', {}).get('uri') or 
                tracking_config.get('dagshub', {}).get('enabled', False) or
                tracking_config.get('mlflow', {}).get('experiment')):
                try:
                    import mlflow
                    if tracking_config.get('mlflow', {}).get('uri'):
                        mlflow.set_tracking_uri(tracking_config['mlflow']['uri'])
                    mlflow.set_experiment(tracking_config.get('mlflow', {}).get('experiment', self.config['project_name']))
                    mlflow_run = mlflow.start_run(run_name=f"ideaweaver-{self.config['project_name']}")
                    if self.verbose:
                        print("✅ MLflow tracking initialized")
                        print(f"   Run ID: {mlflow_run.info.run_id}")
                except ImportError:
                    if self.verbose:
                        print("⚠️  mlflow not installed, skipping MLflow tracking")
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  MLflow setup failed: {str(e)}")
        
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Experiment tracking setup failed: {str(e)}")
        
        return comet_experiment, mlflow_run
    
    def _log_metrics_to_platforms(self, metrics: Dict[str, Any], comet_experiment=None, mlflow_run=None):
        """Log metrics to experiment tracking platforms"""
        if not self.config.get('tracking', {}).get('enabled', False):
            return
        
        try:
            # Prepare metrics for logging
            log_metrics = {}
            
            # Add key metrics
            if 'final_train_loss' in metrics:
                log_metrics['final_train_loss'] = metrics['final_train_loss']
            
            if 'eval_metrics' in metrics:
                eval_metrics = metrics['eval_metrics']
                if 'eval_accuracy' in eval_metrics:
                    log_metrics['overall_accuracy'] = eval_metrics['eval_accuracy']
                    log_metrics['eval_loss'] = eval_metrics.get('eval_loss', 0)
                    log_metrics['eval_f1_macro'] = eval_metrics.get('eval_f1_macro', 0)
            
            # Add training efficiency metrics
            if 'training_stats' in metrics:
                stats = metrics['training_stats']
                log_metrics['train_runtime'] = stats.get('train_runtime', 0)
                log_metrics['train_samples_per_second'] = stats.get('train_samples_per_second', 0)
            
            # Add hyperparameters
            params = self.config.get('params', {})
            log_params = {
                'base_model': self.config.get('base_model', ''),
                'task': self.config.get('task', ''),
                'epochs': params.get('epochs', 0),
                'batch_size': params.get('batch_size', 0),
                'learning_rate': params.get('learning_rate', 0),
                'max_seq_length': params.get('max_seq_length', 0)
            }
            
            # Log to Comet ML
            if comet_experiment:
                try:
                    comet_experiment.log_metrics(log_metrics)
                    comet_experiment.log_parameters(log_params)
                    if self.verbose:
                        print("📊 Metrics logged to Comet ML")
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Failed to log to Comet ML: {str(e)}")
            
            # Log to MLflow
            if mlflow_run:
                try:
                    import mlflow
                    for key, value in log_metrics.items():
                        mlflow.log_metric(key, value)
                    for key, value in log_params.items():
                        mlflow.log_param(key, value)
                    if self.verbose:
                        print("📊 Metrics logged to MLflow")
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Failed to log to MLflow: {str(e)}")
                        
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Experiment logging failed: {str(e)}")
    
    def _log_model_to_platforms(self, model_path: str, comet_experiment=None, mlflow_run=None):
        """Log trained model to experiment tracking platforms"""
        if not self.config.get('tracking', {}).get('enabled', False):
            return
        
        if not os.path.exists(model_path):
            if self.verbose:
                print(f"⚠️  Model path not found: {model_path}")
            return
        
        try:
            # Log to Comet ML
            if comet_experiment:
                try:
                    # Log the entire model directory as artifacts
                    comet_experiment.log_model(
                        name=f"ideaweaver-{self.config['project_name']}",
                        file_or_folder=model_path,
                        metadata={
                            "task": self.config.get('task', ''),
                            "base_model": self.config.get('base_model', ''),
                            "framework": "transformers",
                            "model_type": "text_classification"
                        }
                    )
                    
                    # Also log individual important files
                    important_files = [
                        'config.json', 'model.safetensors', 'tokenizer.json', 
                        'vocab.txt', 'tokenizer_config.json'
                    ]
                    
                    for filename in important_files:
                        filepath = os.path.join(model_path, filename)
                        if os.path.exists(filepath):
                            comet_experiment.log_asset(filepath, file_name=filename)
                    
                    # Register the model in Comet Model Registry if enabled
                    if self.config.get('tracking', {}).get('register_model', False):
                        self._register_model_in_comet_registry(comet_experiment)
                    
                    if self.verbose:
                        print("🤖 Model logged to Comet ML")
                    else:
                        print("🤖 Model pushed to Comet ML")
                        
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Failed to log model to Comet ML: {str(e)}")
                    else:
                        print(f"⚠️  Failed to push model to Comet ML: {str(e)}")
            
            # Log to MLflow
            if mlflow_run:
                try:
                    import mlflow
                    
                    # Check if we should register the model
                    should_register = self.config.get('tracking', {}).get('register_model', False)
                    model_name = f"ideaweaver-{self.config['project_name']}"
                    
                    if should_register:
                        try:
                            # Try to use MLflow's transformers integration
                            mlflow.transformers.log_model(
                                transformers_model=model_path,
                                artifact_path="model",
                                registered_model_name=model_name,
                                task=self.config.get('task', 'text-classification')
                            )
                            
                            if self.verbose:
                                print(f"✅ Model registered in MLflow Model Registry: {model_name}")
                            else:
                                print(f"🏷️  Model registered in Model Registry: {model_name}")
                                
                        except Exception as e:
                            # Fallback: log artifacts and register separately
                            if self.verbose:
                                print(f"⚠️  MLflow transformers integration failed, using fallback: {str(e)}")
                            
                            # Log artifacts first
                            mlflow.log_artifacts(model_path, "model")
                            
                            # Get the run ID from the active run
                            run_id = mlflow.active_run().info.run_id
                            model_uri = f"runs:/{run_id}/model"
                            
                            # Register the model
                            mlflow.register_model(
                                model_uri=model_uri,
                                name=model_name
                            )
                            
                            if self.verbose:
                                print(f"✅ Model registered in MLflow Model Registry (fallback): {model_name}")
                            else:
                                print(f"🏷️  Model registered in Model Registry: {model_name}")
                    else:
                        # Just log artifacts without registration
                        mlflow.log_artifacts(model_path, "model")
                        
                        if self.verbose:
                            print("🤖 Model logged to MLflow")
                        else:
                            print("🤖 Model pushed to MLflow")
                            
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Failed to log model to MLflow: {str(e)}")
                    else:
                        print(f"⚠️  Failed to push model to MLflow: {str(e)}")
                        
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Model logging failed: {str(e)}")
            else:
                print(f"⚠️  Model push failed")
    
    def _finalize_experiment_tracking(self, comet_experiment=None, mlflow_run=None):
        """Clean up experiment tracking"""
        try:
            if comet_experiment:
                comet_experiment.end()
            if mlflow_run:
                import mlflow
                mlflow.end_run()
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Error finalizing experiment tracking: {str(e)}")

    def _register_model_in_comet_registry(self, comet_experiment):
        """Register the model in Comet Model Registry"""
        try:
            # Get the logged model from the experiment
            model_name = f"ideaweaver-{self.config['project_name']}"
            
            if self.verbose:
                print(f"📋 Registering model in Comet Model Registry: {model_name}")
            
            # Register the model using Comet's Model Registry API
            comet_experiment.register_model(
                model_name=model_name,
                version="1.0.0",
                workspace=comet_experiment.workspace,
                tags=[
                    self.config.get('task', 'text_classification'),
                    self.config.get('base_model', '').replace('/', '-'),
                    "ideaweaver",
                    "transformers"
                ],
                description=f"IdeaWeaver trained model based on {self.config.get('base_model', '')} for {self.config.get('task', 'text_classification')}",
                comment=f"Trained with {self.config.get('params', {}).get('epochs', 1)} epochs, batch size {self.config.get('params', {}).get('batch_size', 4)}"
            )
            
            if self.verbose:
                print(f"✅ Model registered in Comet Model Registry: {model_name}")
            else:
                print(f"🏷️  Model registered in Model Registry: {model_name}")
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Failed to register model in Model Registry: {str(e)}")
            else:
                print(f"⚠️  Model registration failed")

    def register_model_in_mlflow(self, mlflow_run):
        """Register the model in the MLflow Model Registry."""
        try:
            import mlflow
            model_path = f"./{self.config['project_name']}"
            
            if not os.path.exists(model_path):
                raise Exception(f"Model path not found: {model_path}")
            
            if self.verbose:
                print(f"📦 Registering model from path: {model_path}")
                print(f"🏷️  Model name: {self.config['project_name']}")
                print(f"🔗 Run ID: {mlflow_run.info.run_id}")
            
            # First try to use MLflow's transformers integration
            try:
                if self.verbose:
                    print("Attempting to register model using MLflow's transformers integration...")
                
                # Log the model using transformers integration
                mlflow.transformers.log_model(
                    transformers_model=model_path,
                    artifact_path="model",
                    registered_model_name=self.config['project_name'],
                    task=self.config.get('task', 'text-classification')
                )
                
                if self.verbose:
                    print(f"✅ Model registered in MLflow Model Registry: {self.config['project_name']}")
                return
                
            except Exception as e:
                if self.verbose:
                    print(f"MLflow transformers integration failed, using fallback: {str(e)}")
            
            # Fallback: Log artifacts and register manually
            if self.verbose:
                print("Using fallback registration method...")
            
            # Log artifacts
            mlflow.log_artifacts(model_path, artifact_path="model")
            
            # Register the model
            model_uri = f"runs:/{mlflow_run.info.run_id}/model"
            mlflow.register_model(
                model_uri=model_uri,
                name=self.config['project_name']
            )
            
            if self.verbose:
                print(f"✅ Model registered in MLflow Model Registry (fallback): {self.config['project_name']}")
            
        except Exception as e:
            error_msg = f"Failed to register model in MLflow: {str(e)}"
            if self.verbose:
                import traceback
                print(f"Detailed error: {traceback.format_exc()}")
            raise Exception(error_msg)

def download_model(model_name: str, save_path: str = './downloaded_model') -> bool:
    """Download a model from Hugging Face Hub"""
    try:
        from transformers import AutoTokenizer, AutoModel
        
        print(f"📥 Downloading model: {model_name}")
        
        # Download tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Save locally
        os.makedirs(save_path, exist_ok=True)
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        
        print(f"✅ Model downloaded to: {save_path}")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {str(e)}")
        return False

def train_model(config_path: str, verbose: bool = False):
    """Train a model using AutoTrain Advanced"""
    
    # Load IdeaWeaver configuration
    config = load_config(config_path)
    
    if verbose:
        print(f"📋 Configuration loaded:")
        for key, value in config.items():
            print(f"   {key}: {value}")
    
    print("🚀 Starting model training...")
    
    # Create project directory for AutoTrain
    project_dir = f"./autotrain_projects/{config['project_name']}"
    os.makedirs(project_dir, exist_ok=True)
    
    # Copy and fix dataset for AutoTrain
    original_dataset = config['dataset']
    train_csv_path = os.path.join(project_dir, 'train.csv')
    
    # Read the original CSV and rename 'label' column to 'target'
    import pandas as pd
    df = pd.read_csv(original_dataset)
    if 'label' in df.columns:
        df = df.rename(columns={'label': 'target'})
    df.to_csv(train_csv_path, index=False)
    
    # Update config to point to project directory
    config['dataset'] = project_dir
    
    # Convert to AutoTrain format
    autotrain_config = create_ideaweaver_config(config)
    
    # Create temporary config file for AutoTrain
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        yaml.dump(autotrain_config, f, default_flow_style=False)
        temp_config_path = f.name
    
    print(f"📝 AutoTrain config written to: {temp_config_path}")
    
    if verbose:
        print("📋 AutoTrain configuration:")
        print(yaml.dump(autotrain_config, default_flow_style=False))
    
    # Run AutoTrain command
    cmd = ['autotrain', '--config', temp_config_path]
    print(f"🔧 Running command: {' '.join(cmd)}")
    
    try:
        # Run the training process
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        print("✅ Training completed successfully!")
        if verbose:
            print("📤 Training output:")
            print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print("❌ Training failed!")
        print(f"Exit code: {e.returncode}")
        if e.stdout:
            print("📤 Output:")
            print(e.stdout)
        if e.stderr:
            print("📤 Error:")
            print(e.stderr)
        return False
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_config_path):
            os.unlink(temp_config_path) 