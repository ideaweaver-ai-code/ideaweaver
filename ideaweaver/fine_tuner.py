"""
Fine-tuning module for LLMs using advanced techniques like LoRA, QLoRA, and full fine-tuning.
Supports instruction following, chat, and completion tasks with comprehensive experiment tracking.
"""

import warnings
import os

# Suppress common ML library warnings
warnings.filterwarnings("ignore", message=".*bitsandbytes.*")
warnings.filterwarnings("ignore", message=".*GPU support.*")
warnings.filterwarnings("ignore", message=".*cadam32bit_grad_fp32.*")
warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")

# Suppress comet_ml auto-logging
os.environ.setdefault("COMET_DISABLE_AUTO_LOGGING", "1")

import json
import torch
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import click

# Core libraries
try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, AutoConfig, 
        TrainingArguments, Trainer, DataCollatorForLanguageModeling,
        BitsAndBytesConfig, get_scheduler
    )
    from datasets import Dataset, load_dataset
    import pandas as pd
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# LoRA libraries
try:
    from peft import (
        LoraConfig, get_peft_model, TaskType, PeftModel,
        prepare_model_for_kbit_training, LoraModel
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# Quantization libraries
try:
    import bitsandbytes as bnb
    BNBITS_AVAILABLE = True
except ImportError:
    BNBITS_AVAILABLE = False

# TRL for instruction tuning
try:
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False


@dataclass
class FineTuningConfig:
    """Configuration for supervised fine-tuning"""
    
    # Model configuration
    model_name: str
    task_type: str = "instruction_following"  # instruction_following, chat, completion, classification
    
    # Fine-tuning method
    method: str = "lora"  # lora, qlora, full, adapter
    
    # LoRA specific parameters
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: Optional[List[str]] = None
    
    # Quantization parameters (for QLoRA)
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    # Training parameters
    output_dir: str = "./fine_tuned_model"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    
    # Data parameters
    max_seq_length: int = 512
    dataset_text_field: str = "text"
    dataset_format: str = "instruction"  # instruction, chat, completion, classification
    
    # Evaluation and saving
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    
    # Logging and tracking
    logging_steps: int = 10
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    
    # Hardware optimization
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    dataloader_pin_memory: bool = False
    
    # Advanced options
    packing: bool = False  # Packing for SFTTrainer
    remove_unused_columns: bool = False
    seed: int = 42


class SupervisedFineTuner:
    """
    Supervised Fine-Tuning Manager with support for LoRA, QLoRA, and full fine-tuning
    """
    
    def __init__(self, config: FineTuningConfig, verbose: bool = False):
        """
        Initialize the fine-tuner
        
        Args:
            config: Fine-tuning configuration
            verbose: Enable verbose logging
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library not available. Install with: pip install transformers")
        
        self.config = config
        self.verbose = verbose
        
        # Setup logging
        logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        if verbose:
            click.echo(f"🚀 Supervised Fine-Tuner initialized")
            click.echo(f"   Model: {config.model_name}")
            click.echo(f"   Method: {config.method}")
            click.echo(f"   Task: {config.task_type}")
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer with appropriate configurations"""
        
        if self.verbose:
            click.echo("🔧 Setting up model and tokenizer...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Setup quantization config for QLoRA
        quantization_config = None
        if self.config.method == "qlora" or self.config.load_in_4bit or self.config.load_in_8bit:
            if not BNBITS_AVAILABLE:
                raise ImportError("bitsandbytes not available. Install with: pip install bitsandbytes")
            
            if self.config.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                    bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
                )
            elif self.config.load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.config.fp16 else (torch.bfloat16 if self.config.bf16 else "auto"),
            "device_map": "auto" if quantization_config else None,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # Resize embeddings if tokenizer was modified
        if len(self.tokenizer) > self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Setup LoRA or QLoRA
        if self.config.method in ["lora", "qlora"]:
            self._setup_lora()
        
        if self.verbose:
            click.echo(f"✅ Model and tokenizer setup complete")
            click.echo(f"   Model parameters: {self.model.num_parameters():,}")
            if hasattr(self.model, 'print_trainable_parameters'):
                self.model.print_trainable_parameters()
    
    def _setup_lora(self):
        """Setup LoRA configuration"""
        
        if not PEFT_AVAILABLE:
            raise ImportError("peft library not available. Install with: pip install peft")
        
        if self.verbose:
            click.echo("🔧 Setting up LoRA configuration...")
        
        # Prepare model for k-bit training if using quantization
        if self.config.method == "qlora" or self.config.load_in_4bit or self.config.load_in_8bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Auto-detect target modules if not specified
        target_modules = self.config.lora_target_modules
        if target_modules is None:
            target_modules = self._find_target_modules()
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        if self.verbose:
            click.echo(f"✅ LoRA setup complete")
            click.echo(f"   Target modules: {target_modules}")
            click.echo(f"   LoRA rank: {self.config.lora_rank}")
            click.echo(f"   LoRA alpha: {self.config.lora_alpha}")
    
    def _find_target_modules(self) -> List[str]:
        """Auto-detect target modules for LoRA"""
        
        # Common target modules for different model architectures
        common_targets = {
            "llama": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "mistral": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "phi": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "gemma": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "qwen": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "gpt2": ["c_attn", "c_proj"],  # GPT-2 style models
            "dialogpt": ["c_attn", "c_proj"],  # DialoGPT uses GPT-2 architecture
            "gpt": ["c_attn", "c_proj"],  # Generic GPT models
            "default": ["c_attn", "c_proj"]  # Default to GPT-2 style since it's more common
        }
        
        model_name_lower = self.config.model_name.lower()
        
        # Check for specific architectures
        for arch_name, targets in common_targets.items():
            if arch_name in model_name_lower:
                return targets
        
        return common_targets["default"]
    
    def prepare_dataset(self, dataset_path: str, test_size: float = 0.1) -> Tuple[Dataset, Optional[Dataset]]:
        """
        Prepare dataset for fine-tuning
        
        Args:
            dataset_path: Path to dataset file
            test_size: Fraction of data to use for evaluation
            
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        
        if self.verbose:
            click.echo(f"📊 Preparing dataset from: {dataset_path}")
        
        # Load dataset
        if dataset_path.endswith('.json') or dataset_path.endswith('.jsonl'):
            if dataset_path.endswith('.jsonl'):
                data = []
                with open(dataset_path, 'r') as f:
                    for line in f:
                        data.append(json.loads(line))
                df = pd.DataFrame(data)
            else:
                df = pd.read_json(dataset_path)
        elif dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
        else:
            # Try loading as HuggingFace dataset
            dataset = load_dataset(dataset_path)
            df = dataset['train'].to_pandas()
        
        # Format dataset based on task type
        if self.config.dataset_format == "instruction":
            df = self._format_instruction_dataset(df)
        elif self.config.dataset_format == "chat":
            df = self._format_chat_dataset(df)
        elif self.config.dataset_format == "completion":
            df = self._format_completion_dataset(df)
        elif self.config.dataset_format == "classification":
            df = self._format_classification_dataset(df)
        
        # Split into train and eval
        if test_size > 0:
            train_df = df.sample(frac=1-test_size, random_state=self.config.seed)
            eval_df = df.drop(train_df.index)
            
            train_dataset = Dataset.from_pandas(train_df)
            eval_dataset = Dataset.from_pandas(eval_df)
        else:
            train_dataset = Dataset.from_pandas(df)
            eval_dataset = None
        
        if self.verbose:
            click.echo(f"✅ Dataset prepared")
            click.echo(f"   Training samples: {len(train_dataset)}")
            if eval_dataset:
                click.echo(f"   Evaluation samples: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset
    
    def _format_instruction_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format dataset for instruction following"""
        
        # Expected columns: instruction, input (optional), output
        formatted_texts = []
        
        for _, row in df.iterrows():
            instruction = row.get('instruction', '')
            input_text = row.get('input', '')
            output = row.get('output', '')
            
            if input_text:
                text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            
            formatted_texts.append(text)
        
        return pd.DataFrame({self.config.dataset_text_field: formatted_texts})
    
    def _format_chat_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format dataset for chat/conversation"""
        
        # Expected columns: messages (list of dict with role and content)
        formatted_texts = []
        
        for _, row in df.iterrows():
            messages = row.get('messages', [])
            formatted_text = ""
            
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                if role == 'user':
                    formatted_text += f"Human: {content}\n\n"
                elif role == 'assistant':
                    formatted_text += f"Assistant: {content}\n\n"
                elif role == 'system':
                    formatted_text = f"System: {content}\n\n" + formatted_text
            
            formatted_texts.append(formatted_text.strip())
        
        return pd.DataFrame({self.config.dataset_text_field: formatted_texts})
    
    def _format_completion_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format dataset for text completion"""
        
        # Expected columns: prompt, completion
        formatted_texts = []
        
        for _, row in df.iterrows():
            prompt = row.get('prompt', '')
            completion = row.get('completion', '')
            
            text = f"{prompt}{completion}"
            formatted_texts.append(text)
        
        return pd.DataFrame({self.config.dataset_text_field: formatted_texts})
    
    def _format_classification_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format dataset for classification tasks"""
        
        # Expected columns: text, label
        formatted_texts = []
        
        for _, row in df.iterrows():
            text = row.get('text', '')
            label = row.get('label', '')
            
            formatted_text = f"Text: {text}\nLabel: {label}"
            formatted_texts.append(formatted_text)
        
        return pd.DataFrame({self.config.dataset_text_field: formatted_texts})
    
    def setup_trainer(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """Setup the trainer for fine-tuning"""
        
        if self.verbose:
            click.echo("🔧 Setting up trainer...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            max_grad_norm=self.config.max_grad_norm,
            eval_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps if eval_dataset else None,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end and eval_dataset is not None,
            metric_for_best_model=self.config.metric_for_best_model,
            logging_steps=self.config.logging_steps,
            report_to=self.config.report_to,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            remove_unused_columns=self.config.remove_unused_columns,
            seed=self.config.seed,
        )
        
        # Use SFTTrainer if available (better for instruction tuning)
        if TRL_AVAILABLE and self.config.task_type in ["instruction_following", "chat"]:
            # For now, use standard Trainer due to TRL API changes
            # TODO: Update to new SFTTrainer API when stable
            use_standard_trainer = True
        else:
            use_standard_trainer = True
            
        if use_standard_trainer:
            # Use standard Trainer with data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )
            
            # Tokenize datasets
            def tokenize_function(examples):
                # Handle both single examples and batches
                texts = examples[self.config.dataset_text_field]
                if isinstance(texts, str):
                    texts = [texts]
                
                # Tokenize with proper padding and truncation
                tokenized = self.tokenizer(
                    texts,
                    truncation=True,
                    padding='max_length',  # Use max_length padding for consistent batch sizes
                    max_length=self.config.max_seq_length,
                    return_tensors=None,  # Return lists, not tensors
                )
                
                # For causal LM, labels are the same as input_ids
                tokenized["labels"] = tokenized["input_ids"].copy()
                
                return tokenized
            
            train_dataset = train_dataset.map(tokenize_function, batched=True)
            if eval_dataset:
                eval_dataset = eval_dataset.map(tokenize_function, batched=True)
            
            # Remove original text columns to avoid collation issues
            columns_to_remove = [col for col in train_dataset.column_names if col not in ['input_ids', 'attention_mask', 'labels']]
            if columns_to_remove:
                train_dataset = train_dataset.remove_columns(columns_to_remove)
                if eval_dataset:
                    eval_dataset = eval_dataset.remove_columns(columns_to_remove)
            
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
            )
        else:
            # Original SFTTrainer code (kept for reference)
            # In TRL 0.13.0, SFTTrainer has a much simpler API
            self.trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=self.tokenizer,  # Use processing_class instead of tokenizer
            )
        
        if self.verbose:
            click.echo("✅ Trainer setup complete")
    
    def train(self):
        """Start the fine-tuning process"""
        
        if self.verbose:
            click.echo("🚀 Starting fine-tuning...")
        
        # Start training
        train_result = self.trainer.train()
        
        # Save the final model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        if self.verbose:
            click.echo("✅ Fine-tuning completed!")
            click.echo(f"📁 Model saved to: {self.config.output_dir}")
        
        return train_result
    
    def save_model(self, save_path: Optional[str] = None):
        """Save the fine-tuned model"""
        
        save_path = save_path or self.config.output_dir
        
        if self.config.method in ["lora", "qlora"]:
            # Save LoRA adapters
            self.model.save_pretrained(save_path)
        else:
            # Save full model
            self.trainer.save_model(save_path)
        
        # Always save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        # Save configuration
        config_path = os.path.join(save_path, "fine_tuning_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        if self.verbose:
            click.echo(f"💾 Model saved to: {save_path}")
    
    def evaluate(self, eval_dataset: Dataset = None) -> Dict[str, float]:
        """Evaluate the fine-tuned model"""
        
        if eval_dataset is None and hasattr(self.trainer, 'eval_dataset'):
            eval_dataset = self.trainer.eval_dataset
        
        if eval_dataset is None:
            if self.verbose:
                click.echo("⚠️  No evaluation dataset available")
            return {}
        
        if self.verbose:
            click.echo("📊 Evaluating model...")
        
        eval_results = self.trainer.evaluate(eval_dataset=eval_dataset)
        
        if self.verbose:
            click.echo("✅ Evaluation complete")
            for key, value in eval_results.items():
                click.echo(f"   {key}: {value:.4f}")
        
        return eval_results
    
    @classmethod
    def load_fine_tuned_model(cls, model_path: str, base_model: Optional[str] = None):
        """Load a fine-tuned model"""
        
        # Load configuration
        config_path = os.path.join(model_path, "fine_tuning_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = FineTuningConfig(**config_dict)
        else:
            # Fallback configuration
            config = FineTuningConfig(
                model_name=base_model or "microsoft/DialoGPT-medium",
                method="lora"
            )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model
        if config.method in ["lora", "qlora"]:
            # Load base model first
            base_model_name = base_model or config.model_name
            model = AutoModelForCausalLM.from_pretrained(base_model_name)
            
            # Load LoRA adapters
            model = PeftModel.from_pretrained(model, model_path)
        else:
            # Load full fine-tuned model
            model = AutoModelForCausalLM.from_pretrained(model_path)
        
        return model, tokenizer, config


def create_fine_tuning_config(
    model_name: str,
    method: str = "lora",
    task_type: str = "instruction_following",
    **kwargs
) -> FineTuningConfig:
    """
    Create a fine-tuning configuration with sensible defaults
    
    Args:
        model_name: Base model name
        method: Fine-tuning method (lora, qlora, full)
        task_type: Task type (instruction_following, chat, completion, classification)
        **kwargs: Additional configuration parameters
        
    Returns:
        FineTuningConfig instance
    """
    
    # Default configurations for different methods
    method_defaults = {
        "lora": {
            "lora_rank": 16,
            "lora_alpha": 32,
            "learning_rate": 2e-4,
            "per_device_train_batch_size": 4,
        },
        "qlora": {
            "load_in_4bit": True,
            "lora_rank": 64,
            "lora_alpha": 16,
            "learning_rate": 2e-4,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
        },
        "full": {
            "learning_rate": 5e-5,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "gradient_checkpointing": True,
        }
    }
    
    # Apply method-specific defaults
    config_params = method_defaults.get(method, {})
    config_params.update(kwargs)
    
    return FineTuningConfig(
        model_name=model_name,
        method=method,
        task_type=task_type,
        **config_params
    ) 