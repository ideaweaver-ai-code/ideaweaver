"""
Docker Manager for IdeaWeaver
Handles Docker image building and management for trained models
"""

import os
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
import click


class DockerManager:
    """Manages Docker operations for trained models"""
    
    def __init__(self, model_path: str, verbose: bool = False):
        """
        Initialize Docker manager
        
        Args:
            model_path: Path to the trained model directory
            verbose: Enable verbose logging
        """
        self.model_path = Path(model_path)
        self.verbose = verbose
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        # Validate model directory structure
        self._validate_model_directory()
    
    def _validate_model_directory(self):
        """Validate that the model directory contains required files"""
        required_files = ['config.json']
        optional_files = ['model.safetensors', 'pytorch_model.bin', 'tokenizer.json', 'tokenizer_config.json']
        
        missing_required = []
        for file in required_files:
            if not (self.model_path / file).exists():
                missing_required.append(file)
        
        if missing_required:
            raise FileNotFoundError(f"Missing required model files: {missing_required}")
        
        # Check for at least one model file
        model_files = [f for f in optional_files if (self.model_path / f).exists()]
        if not model_files:
            click.echo("⚠️  Warning: No model weight files found. This might be a LoRA adapter only.")
    
    def build_docker_image(self, 
                          image_name: str,
                          base_image: str = "python:3.9-slim",
                          port: int = 8000,
                          requirements: Optional[List[str]] = None,
                          custom_dockerfile: Optional[str] = None,
                          verbose: Optional[bool] = None) -> bool:
        """
        Build Docker image for the trained model
        
        Args:
            image_name: Name for the Docker image
            base_image: Base Docker image to use
            port: Port to expose in the container
            requirements: Additional Python packages to install
            custom_dockerfile: Path to custom Dockerfile template
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Use provided verbose setting or fall back to instance setting
            use_verbose = verbose if verbose is not None else self.verbose
            
            if use_verbose:
                click.echo(f"🐳 Building Docker image: {image_name}")
                click.echo(f"   Model path: {self.model_path}")
                click.echo(f"   Base image: {base_image}")
                click.echo(f"   Port: {port}")
            
            # Create temporary directory for Docker build context
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Copy model files to temp directory
                model_dest = temp_path / "model"
                shutil.copytree(self.model_path, model_dest)
                
                # Generate Dockerfile
                dockerfile_content = self._generate_dockerfile(
                    base_image=base_image,
                    port=port,
                    requirements=requirements,
                    custom_dockerfile=custom_dockerfile
                )
                
                # Write Dockerfile
                dockerfile_path = temp_path / "Dockerfile"
                with open(dockerfile_path, 'w') as f:
                    f.write(dockerfile_content)
                
                # Create inference server script
                self._create_inference_server(temp_path, port)
                
                # Create requirements.txt
                self._create_requirements_file(temp_path, requirements)
                
                # Build Docker image
                cmd = [
                    'docker', 'build',
                    '-t', image_name,
                    '.',
                ]
                
                if use_verbose:
                    click.echo(f"🔧 Running: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    cwd=temp_path,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    click.echo(f"✅ Docker image built successfully: {image_name}")
                    if use_verbose and result.stdout:
                        click.echo(result.stdout)
                    return True
                else:
                    click.echo(f"❌ Docker build failed:")
                    if result.stderr:
                        click.echo(result.stderr)
                    return False
                    
        except Exception as e:
            click.echo(f"❌ Error building Docker image: {str(e)}")
            return False
    
    def _generate_dockerfile(self, 
                           base_image: str,
                           port: int,
                           requirements: Optional[List[str]] = None,
                           custom_dockerfile: Optional[str] = None) -> str:
        """Generate Dockerfile content"""
        
        if custom_dockerfile and os.path.exists(custom_dockerfile):
            with open(custom_dockerfile, 'r') as f:
                return f.read()
        
        dockerfile = f"""FROM {base_image}

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY model/ ./model/

# Copy inference server
COPY inference_server.py .

# Expose port
EXPOSE {port}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{port}/health || exit 1

# Run inference server
CMD ["python", "inference_server.py"]
"""
        return dockerfile
    
    def _create_inference_server(self, temp_path: Path, port: int):
        """Create FastAPI inference server"""
        
        server_code = f'''#!/usr/bin/env python3
"""
FastAPI Inference Server for IdeaWeaver Model
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# Request/Response models
class InferenceRequest(BaseModel):
    text: str
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    num_return_sequences: Optional[int] = 1

class InferenceResponse(BaseModel):
    generated_text: List[str]
    model_info: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str

# Global model and tokenizer
model = None
tokenizer = None
device = None
model_info = {{}}

app = FastAPI(
    title="IdeaWeaver Model API",
    description="FastAPI server for trained model inference",
    version="1.0.0"
)

def load_model():
    """Load model and tokenizer"""
    global model, tokenizer, device, model_info
    
    model_path = "./model"
    
    try:
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        if device == "cpu":
            model = model.to(device)
        
        model.eval()
        
        # Load model info
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                model_info = {{
                    "model_type": config_data.get("model_type", "unknown"),
                    "vocab_size": config_data.get("vocab_size", 0),
                    "hidden_size": config_data.get("hidden_size", 0),
                    "num_hidden_layers": config_data.get("num_hidden_layers", 0),
                    "device": device
                }}
        
        print(f"✅ Model loaded successfully on {{device}}")
        
    except Exception as e:
        print(f"❌ Error loading model: {{str(e)}}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=device or "unknown"
    )

@app.get("/info")
async def model_info_endpoint():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {{
        "model_info": model_info,
        "status": "ready"
    }}

@app.post("/generate", response_model=InferenceResponse)
async def generate_text(request: InferenceRequest):
    """Generate text using the loaded model"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Tokenize input
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to device
        inputs = {{k: v.to(device) for k, v in inputs.items()}}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                num_return_sequences=request.num_return_sequences,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode generated text
        generated_texts = []
        for output in outputs:
            # Remove input tokens from output
            generated = output[inputs['input_ids'].shape[1]:]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            generated_texts.append(text)
        
        return InferenceResponse(
            generated_text=generated_texts,
            model_info=model_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {{str(e)}}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {{
        "message": "IdeaWeaver Model API",
        "status": "running",
        "endpoints": ["/health", "/info", "/generate", "/docs"]
    }}

if __name__ == "__main__":
    uvicorn.run(
        "inference_server:app",
        host="0.0.0.0",
        port={port},
        reload=False
    )
'''
        
        with open(temp_path / "inference_server.py", 'w') as f:
            f.write(server_code)
    
    def _create_requirements_file(self, temp_path: Path, additional_requirements: Optional[List[str]] = None):
        """Create requirements.txt file"""
        
        base_requirements = [
            "torch>=1.9.0",
            "transformers>=4.20.0",
            "fastapi>=0.68.0",
            "uvicorn[standard]>=0.15.0",
            "pydantic>=1.8.0",
            "accelerate>=0.12.0",
            "sentencepiece>=0.1.97",
            "protobuf>=3.20.0"
        ]
        
        if additional_requirements:
            base_requirements.extend(additional_requirements)
        
        with open(temp_path / "requirements.txt", 'w') as f:
            f.write('\n'.join(base_requirements))
    
    def list_images(self) -> List[Dict[str, str]]:
        """List Docker images related to IdeaWeaver models"""
        try:
            result = subprocess.run(
                ['docker', 'images', '--filter', 'label=ideaweaver=true', '--format', 'json'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                images = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        try:
                            images.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                return images
            else:
                return []
                
        except Exception as e:
            if self.verbose:
                click.echo(f"Error listing images: {str(e)}")
            return []
    
    def remove_image(self, image_name: str) -> bool:
        """Remove a Docker image"""
        try:
            result = subprocess.run(
                ['docker', 'rmi', image_name],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                click.echo(f"✅ Removed Docker image: {image_name}")
                return True
            else:
                click.echo(f"❌ Failed to remove image: {result.stderr}")
                return False
                
        except Exception as e:
            click.echo(f"❌ Error removing image: {str(e)}")
            return False
    
    def run_container(self, 
                     image_name: str,
                     container_name: Optional[str] = None,
                     port_mapping: str = "8000:8000",
                     detached: bool = True) -> bool:
        """Run a Docker container from the built image"""
        try:
            cmd = ['docker', 'run']
            
            if detached:
                cmd.append('-d')
            
            if container_name:
                cmd.extend(['--name', container_name])
            
            cmd.extend(['-p', port_mapping])
            cmd.append(image_name)
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                click.echo(f"✅ Container started successfully")
                if self.verbose and result.stdout:
                    click.echo(f"Container ID: {result.stdout.strip()}")
                return True
            else:
                click.echo(f"❌ Failed to start container: {result.stderr}")
                return False
                
        except Exception as e:
            click.echo(f"❌ Error running container: {str(e)}")
            return False


def build_model_docker_image(model_path: str, 
                           image_name: str,
                           **kwargs) -> bool:
    """
    Convenience function to build Docker image for a trained model
    
    Args:
        model_path: Path to the trained model directory
        image_name: Name for the Docker image
        **kwargs: Additional arguments for DockerManager.build_docker_image()
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        manager = DockerManager(model_path, verbose=kwargs.get('verbose', False))
        return manager.build_docker_image(image_name, **kwargs)
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")
        return False 