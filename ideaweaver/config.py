import yaml
import os
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate YAML configuration file"""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['project_name', 'task', 'base_model', 'dataset']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
    
    # Set defaults
    defaults = {
        'backend': 'local',
        'method': 'sft',
        'hub': {'push_to_hub': False},
        'params': {
            'epochs': 3,
            'batch_size': 8,
            'learning_rate': 2e-5,
            'max_seq_length': 128,
        }
    }
    
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if sub_key not in config[key]:
                    config[key][sub_key] = sub_value
    
    return config

def create_ideaweaver_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert IdeaWeaver config to training format"""
    
    ideaweaver_config = {
        'task': config['task'],
        'base_model': config['base_model'],
        'project_name': config['project_name'],
        'log': 'tensorboard',
        'data': {
            'path': config['dataset'],
            'train_split': 'train',
            'valid_split': None,
            'column_mapping': {
                'text': 'text',
                'target': 'target'
            }
        },
        'params': {
            'epochs': config['params']['epochs'],
            'batch_size': config['params']['batch_size'],
            'lr': config['params']['learning_rate'],
            'max_seq_length': config['params']['max_seq_length'],
        },
        'backend': config['backend'],
    }
    
    # Add hub configuration if specified
    if config['hub']['push_to_hub']:
        ideaweaver_config['hub'] = config['hub']
    
    return ideaweaver_config 