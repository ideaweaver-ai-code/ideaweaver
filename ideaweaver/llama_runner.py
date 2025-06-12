"""
Llama.cpp Integration for IdeaWeaver
Fast inference engine for running fine-tuned and quantized models
"""

import os
import json
import subprocess
import tempfile
import logging
from typing import Dict, Any, Optional, List, Union, Iterator
from dataclasses import dataclass
from pathlib import Path
import click

# Core libraries
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Llama.cpp Python bindings
try:
    import llama_cpp
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False


@dataclass
class LlamaConfig:
    """Configuration for llama.cpp inference"""
    
    # Model settings
    model_path: str
    n_ctx: int = 2048
    n_batch: int = 512
    n_threads: int = -1  # -1 for auto-detect
    
    # Generation settings
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 256
    
    # Performance settings
    use_mlock: bool = False
    use_mmap: bool = True
    n_gpu_layers: int = 0  # 0 for CPU-only
    
    # Server settings (for API mode)
    host: str = "127.0.0.1"
    port: int = 8080
    

@dataclass
class LlamaResponse:
    """Response from llama.cpp"""
    
    text: str
    tokens_generated: int
    tokens_per_second: float
    total_time: float
    metadata: Dict[str, Any]


class LlamaRunner:
    """
    Llama.cpp Integration for fast model inference
    Supports both Python bindings and server mode
    """
    
    def __init__(self, config: LlamaConfig, verbose: bool = False):
        """
        Initialize the Llama runner
        
        Args:
            config: Llama configuration
            verbose: Enable verbose logging
        """
        self.config = config
        self.verbose = verbose
        
        # Setup logging
        logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self.model = None
        self.server_process = None
        
        if verbose:
            click.echo(f"🦙 Llama Runner initialized")
            click.echo(f"   Model: {config.model_path}")
            click.echo(f"   Context: {config.n_ctx}")
            click.echo(f"   GPU layers: {config.n_gpu_layers}")
    
    def load_model(self, mode: str = "python") -> bool:
        """
        Load the model for inference
        
        Args:
            mode: Loading mode ("python" or "server")
            
        Returns:
            True if successful
        """
        
        if mode == "python":
            return self._load_python_model()
        elif mode == "server":
            return self._start_server()
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    
    def _load_python_model(self) -> bool:
        """Load model using Python bindings"""
        
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python not available. Install with: pip install llama-cpp-python")
        
        if self.verbose:
            click.echo("🔄 Loading model with Python bindings...")
        
        try:
            # Determine number of threads
            n_threads = self.config.n_threads
            if n_threads == -1:
                import multiprocessing
                n_threads = multiprocessing.cpu_count()
            
            # Load model
            self.model = llama_cpp.Llama(
                model_path=self.config.model_path,
                n_ctx=self.config.n_ctx,
                n_batch=self.config.n_batch,
                n_threads=n_threads,
                use_mlock=self.config.use_mlock,
                use_mmap=self.config.use_mmap,
                n_gpu_layers=self.config.n_gpu_layers,
                verbose=self.verbose
            )
            
            if self.verbose:
                click.echo("✅ Model loaded successfully!")
            
            return True
            
        except Exception as e:
            if self.verbose:
                click.echo(f"❌ Failed to load model: {e}")
            return False
    
    def _start_server(self) -> bool:
        """Start llama.cpp server"""
        
        if self.verbose:
            click.echo("🚀 Starting llama.cpp server...")
        
        # Find llama.cpp server binary
        server_path = self._find_server_binary()
        if not server_path:
            if self.verbose:
                click.echo("❌ llama.cpp server binary not found")
            return False
        
        # Build server command
        cmd = [
            server_path,
            "-m", self.config.model_path,
            "-c", str(self.config.n_ctx),
            "-b", str(self.config.n_batch),
            "--host", self.config.host,
            "--port", str(self.config.port),
        ]
        
        if self.config.n_gpu_layers > 0:
            cmd.extend(["-ngl", str(self.config.n_gpu_layers)])
        
        if self.config.n_threads > 0:
            cmd.extend(["-t", str(self.config.n_threads)])
        
        try:
            # Start server process
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE if not self.verbose else None,
                stderr=subprocess.PIPE if not self.verbose else None
            )
            
            # Wait a moment for server to start
            import time
            time.sleep(3)
            
            # Check if server is running
            if self._check_server_health():
                if self.verbose:
                    click.echo(f"✅ Server started on {self.config.host}:{self.config.port}")
                return True
            else:
                if self.verbose:
                    click.echo("❌ Server failed to start properly")
                return False
                
        except Exception as e:
            if self.verbose:
                click.echo(f"❌ Failed to start server: {e}")
            return False
    
    def _find_server_binary(self) -> Optional[str]:
        """Find llama.cpp server binary"""
        
        # Common server binary names and locations
        possible_paths = [
            "server",
            "llama-server",
            "./server",
            "./llama-server",
            "/usr/local/bin/server",
            "/usr/local/bin/llama-server",
            os.path.expanduser("~/llama.cpp/server"),
            os.path.expanduser("~/llama.cpp/llama-server"),
            os.path.expanduser("~/llama.cpp/build/bin/server"),
            os.path.expanduser("~/llama.cpp/build/bin/llama-server"),
        ]
        
        for path in possible_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        
        return None
    
    def _check_server_health(self) -> bool:
        """Check if server is healthy"""
        
        if not REQUESTS_AVAILABLE:
            return True  # Assume healthy if we can't check
        
        try:
            response = requests.get(
                f"http://{self.config.host}:{self.config.port}/health",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def generate(self, prompt: str, **kwargs) -> LlamaResponse:
        """
        Generate text using the loaded model
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Returns:
            LlamaResponse with generated text and metadata
        """
        
        if self.model is not None:
            return self._generate_python(prompt, **kwargs)
        elif self.server_process is not None:
            return self._generate_server(prompt, **kwargs)
        else:
            raise RuntimeError("No model loaded. Call load_model() first.")
    
    def _generate_python(self, prompt: str, **kwargs) -> LlamaResponse:
        """Generate using Python bindings"""
        
        import time
        start_time = time.time()
        
        # Merge generation parameters
        gen_params = {
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "top_k": kwargs.get("top_k", self.config.top_k),
            "repeat_penalty": kwargs.get("repeat_penalty", self.config.repeat_penalty),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        
        # Generate
        output = self.model(prompt, **gen_params)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Extract response
        generated_text = output['choices'][0]['text']
        tokens_generated = output['usage']['completion_tokens']
        tokens_per_second = tokens_generated / total_time if total_time > 0 else 0
        
        return LlamaResponse(
            text=generated_text,
            tokens_generated=tokens_generated,
            tokens_per_second=tokens_per_second,
            total_time=total_time,
            metadata=output
        )
    
    def _generate_server(self, prompt: str, **kwargs) -> LlamaResponse:
        """Generate using server API"""
        
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library not available. Install with: pip install requests")
        
        import time
        start_time = time.time()
        
        # Prepare request
        gen_params = {
            "prompt": prompt,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "top_k": kwargs.get("top_k", self.config.top_k),
            "repeat_penalty": kwargs.get("repeat_penalty", self.config.repeat_penalty),
            "n_predict": kwargs.get("max_tokens", self.config.max_tokens),
        }
        
        # Send request
        response = requests.post(
            f"http://{self.config.host}:{self.config.port}/completion",
            json=gen_params,
            timeout=60
        )
        
        response.raise_for_status()
        result = response.json()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Extract response
        generated_text = result.get('content', '')
        tokens_generated = result.get('tokens_predicted', 0)
        tokens_per_second = tokens_generated / total_time if total_time > 0 else 0
        
        return LlamaResponse(
            text=generated_text,
            tokens_generated=tokens_generated,
            tokens_per_second=tokens_per_second,
            total_time=total_time,
            metadata=result
        )
    
    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """
        Generate text with streaming (Python bindings only)
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Yields:
            Generated text chunks
        """
        
        if self.model is None:
            raise RuntimeError("Streaming only available with Python bindings")
        
        # Merge generation parameters
        gen_params = {
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "top_k": kwargs.get("top_k", self.config.top_k),
            "repeat_penalty": kwargs.get("repeat_penalty", self.config.repeat_penalty),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": True,
        }
        
        # Generate with streaming
        for output in self.model(prompt, **gen_params):
            chunk = output['choices'][0]['text']
            yield chunk
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LlamaResponse:
        """
        Generate response for chat conversation
        
        Args:
            messages: List of chat messages with 'role' and 'content'
            **kwargs: Generation parameters
            
        Returns:
            LlamaResponse with generated text
        """
        
        # Convert messages to prompt format
        prompt = self._format_chat_prompt(messages)
        
        return self.generate(prompt, **kwargs)
    
    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into a prompt"""
        
        formatted_messages = []
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                formatted_messages.append(f"System: {content}")
            elif role == 'user':
                formatted_messages.append(f"Human: {content}")
            elif role == 'assistant':
                formatted_messages.append(f"Assistant: {content}")
        
        # Add assistant prompt at the end
        formatted_messages.append("Assistant:")
        
        return "\n\n".join(formatted_messages)
    
    def benchmark(self, prompt: str = "Tell me about artificial intelligence.", runs: int = 5) -> Dict[str, float]:
        """
        Benchmark model performance
        
        Args:
            prompt: Test prompt
            runs: Number of benchmark runs
            
        Returns:
            Performance metrics
        """
        
        if self.verbose:
            click.echo(f"🏃 Running benchmark ({runs} runs)...")
        
        times = []
        tokens_per_second_list = []
        
        for i in range(runs):
            if self.verbose:
                click.echo(f"   Run {i+1}/{runs}")
            
            response = self.generate(prompt, max_tokens=50)
            times.append(response.total_time)
            tokens_per_second_list.append(response.tokens_per_second)
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        avg_tokens_per_second = sum(tokens_per_second_list) / len(tokens_per_second_list)
        min_time = min(times)
        max_time = max(times)
        
        results = {
            "average_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "average_tokens_per_second": avg_tokens_per_second,
            "runs": runs
        }
        
        if self.verbose:
            click.echo(f"📊 Benchmark Results:")
            click.echo(f"   Average time: {avg_time:.2f}s")
            click.echo(f"   Average tokens/s: {avg_tokens_per_second:.1f}")
            click.echo(f"   Min time: {min_time:.2f}s")
            click.echo(f"   Max time: {max_time:.2f}s")
        
        return results
    
    def cleanup(self):
        """Clean up resources"""
        
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.server_process is not None:
            self.server_process.terminate()
            self.server_process.wait()
            self.server_process = None
        
        if self.verbose:
            click.echo("🧹 Cleanup completed")


def create_llama_config(
    model_path: str,
    context_length: int = 2048,
    gpu_layers: int = 0,
    **kwargs
) -> LlamaConfig:
    """
    Create a llama configuration with sensible defaults
    
    Args:
        model_path: Path to GGUF model file
        context_length: Context window size
        gpu_layers: Number of GPU layers (0 for CPU-only)
        **kwargs: Additional configuration parameters
        
    Returns:
        LlamaConfig instance
    """
    
    return LlamaConfig(
        model_path=model_path,
        n_ctx=context_length,
        n_gpu_layers=gpu_layers,
        **kwargs
    )


def install_llama_cpp():
    """Install llama-cpp-python with appropriate backend"""
    
    import subprocess
    import sys
    
    click.echo("🔧 Installing llama-cpp-python...")
    
    # Detect if CUDA is available
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    
    # Choose appropriate installation command
    if has_cuda:
        # Install with CUDA support
        cmd = [sys.executable, "-m", "pip", "install", "llama-cpp-python[cuda]"]
        click.echo("   Installing with CUDA support...")
    else:
        # Install CPU-only version
        cmd = [sys.executable, "-m", "pip", "install", "llama-cpp-python"]
        click.echo("   Installing CPU-only version...")
    
    try:
        subprocess.run(cmd, check=True)
        click.echo("✅ llama-cpp-python installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Installation failed: {e}")
        return False 