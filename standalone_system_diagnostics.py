#!/usr/bin/env python3
"""
Standalone System Diagnostics Agent
A minimal implementation for testing system diagnostic functionality without full IdeaWeaver installation.

Dependencies:
pip install crewai langchain-openai requests

Usage:
python standalone_system_diagnostics.py [--openai-api-key YOUR_KEY] [--verbose]
"""

import subprocess
import platform
import argparse
import sys
import os
from typing import Dict, Optional, Tuple
import warnings
import requests

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def setup_intelligent_llm(openai_api_key: Optional[str] = None) -> Tuple:
    """
    Intelligently set up LLM with Ollama as first preference, OpenAI as fallback.
    Returns (llm_instance, llm_type, model_used)
    """
    
    # First try Ollama
    try:
        print("🔍 Checking for Ollama availability...")
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            models_data = response.json()
            available_models = [model['name'] for model in models_data.get('models', [])]
            
            if available_models:
                # Use the first available model
                model_to_use = available_models[0]
                print(f"✅ Ollama is available! Using model: {model_to_use}")
                
                try:
                    from crewai import LLM
                    llm = LLM(
                        model=f"ollama/{model_to_use}",
                        base_url="http://localhost:11434"
                    )
                    print(f"✅ Created CrewAI LLM wrapper successfully")
                    return llm, 'ollama', model_to_use
                except Exception as e:
                    print(f"⚠️ CrewAI LLM wrapper failed: {e}")
                    print("🔄 Falling back to OpenAI...")
            else:
                print("⚠️ Ollama is running but no models are available")
                print("💡 Try: ollama pull phi3:mini")
        else:
            print("⚠️ Ollama is not responding properly")
            
    except Exception as e:
        print(f"⚠️ Ollama check failed: {e}")
    
    # Fallback to OpenAI
    print("🔄 Setting up OpenAI...")
    
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    elif not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OpenAI API key required. Please provide --openai-api-key parameter or set OPENAI_API_KEY environment variable.")
    
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
        print("✅ OpenAI setup successful")
        return llm, 'openai', 'gpt-4o-mini'
    except Exception as e:
        print(f"❌ OpenAI setup failed: {e}")
        raise RuntimeError("Both Ollama and OpenAI failed")


class StandaloneSystemDiagnostics:
    """
    Standalone System Diagnostic tool with real command execution
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the System Diagnostic tool"""
        self.llm, self.llm_type, self.model_name = setup_intelligent_llm(openai_api_key)
        print(f"🔧 System Diagnostics initialized with {self.llm_type} ({self.model_name})")
    
    def _execute_system_commands(self) -> Dict[str, str]:
        """Execute actual system diagnostic commands and return their output"""
        results = {}
        system = platform.system()
        
        try:
            # 1. CPU & Load Analysis
            print("🖥️  Executing CPU & Load analysis...")
            try:
                result = subprocess.run(['w'], capture_output=True, text=True, timeout=10)
                results['cpu_load'] = {
                    'command': 'w',
                    'output': result.stdout if result.returncode == 0 else result.stderr,
                    'success': result.returncode == 0
                }
            except Exception as e:
                results['cpu_load'] = {
                    'command': 'w',
                    'output': f"Error executing 'w': {str(e)}",
                    'success': False
                }
            
            # 2. Memory Analysis
            print("🧠 Executing Memory analysis...")
            if system == "Darwin":  # macOS
                try:
                    result = subprocess.run(['vm_stat'], capture_output=True, text=True, timeout=10)
                    results['memory'] = {
                        'command': 'vm_stat',
                        'output': result.stdout if result.returncode == 0 else result.stderr,
                        'success': result.returncode == 0
                    }
                except Exception as e:
                    results['memory'] = {
                        'command': 'vm_stat',
                        'output': f"Error executing 'vm_stat': {str(e)}",
                        'success': False
                    }
            else:  # Linux
                try:
                    result = subprocess.run(['free', '-m'], capture_output=True, text=True, timeout=10)
                    results['memory'] = {
                        'command': 'free -m',
                        'output': result.stdout if result.returncode == 0 else result.stderr,
                        'success': result.returncode == 0
                    }
                except Exception as e:
                    results['memory'] = {
                        'command': 'free -m',
                        'output': f"Error executing 'free -m': {str(e)}",
                        'success': False
                    }
            
            # 3. Network Analysis
            print("🌐 Executing Network analysis...")
            network_commands = []
            if system == "Darwin":  # macOS
                network_commands = [
                    (['ifconfig'], 'ifconfig'),
                    (['netstat', '-rn'], 'netstat -rn')
                ]
            else:  # Linux
                network_commands = [
                    (['ip', 'addr', 'show'], 'ip addr show'),
                    (['ss', '-tuln'], 'ss -tuln')
                ]
            
            network_output = []
            for cmd, cmd_str in network_commands:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        network_output.append(f"=== {cmd_str} ===\n{result.stdout}")
                    else:
                        network_output.append(f"=== {cmd_str} (ERROR) ===\n{result.stderr}")
                except Exception as e:
                    network_output.append(f"=== {cmd_str} (FAILED) ===\nError: {str(e)}")
            
            results['network'] = {
                'command': 'Network analysis',
                'output': '\n\n'.join(network_output),
                'success': len(network_output) > 0
            }
            
            # 4. Process Analysis
            print("⚡ Executing Process analysis...")
            try:
                if system == "Darwin":  # macOS
                    result = subprocess.run(['top', '-l', '1', '-n', '10'], capture_output=True, text=True, timeout=15)
                else:  # Linux
                    result = subprocess.run(['top', '-b', '-n1'], capture_output=True, text=True, timeout=15)
                
                results['processes'] = {
                    'command': 'top -l 1 -n 10' if system == "Darwin" else 'top -b -n1',
                    'output': result.stdout if result.returncode == 0 else result.stderr,
                    'success': result.returncode == 0
                }
            except Exception as e:
                results['processes'] = {
                    'command': 'top',
                    'output': f"Error executing 'top': {str(e)}",
                    'success': False
                }
            
            # 5. System Info (macOS)
            if system == "Darwin":
                print("🍎 Executing macOS-specific diagnostics...")
                try:
                    result = subprocess.run(['sysctl', '-n', 'hw.ncpu'], capture_output=True, text=True, timeout=5)
                    cpu_cores = result.stdout.strip() if result.returncode == 0 else "unknown"
                    
                    result = subprocess.run(['uptime'], capture_output=True, text=True, timeout=5)
                    uptime_output = result.stdout.strip() if result.returncode == 0 else "unknown"
                    
                    results['system_info'] = {
                        'command': 'System Information',
                        'output': f"CPU Cores: {cpu_cores}\nUptime: {uptime_output}",
                        'success': True
                    }
                except Exception as e:
                    results['system_info'] = {
                        'command': 'System Information',
                        'output': f"Error getting system info: {str(e)}",
                        'success': False
                    }
            
        except Exception as e:
            results['error'] = {
                'command': 'System Diagnostics',
                'output': f"General error: {str(e)}",
                'success': False
            }
        
        return results
    
    def run_diagnostics(self, verbose: bool = False) -> Dict:
        """Run system diagnostics and get AI recommendations"""
        try:
            from crewai import Agent, Task, Crew, Process
            
            # Execute actual system commands first
            command_results = self._execute_system_commands()
            
            # Prepare diagnostic data for the agent
            diagnostic_data = "=== REAL SYSTEM DIAGNOSTIC OUTPUT ===\n\n"
            for section, data in command_results.items():
                diagnostic_data += f"📋 {section.upper()} DIAGNOSTICS:\n"
                diagnostic_data += f"Command: {data['command']}\n"
                diagnostic_data += f"Status: {'SUCCESS' if data['success'] else 'FAILED'}\n"
                diagnostic_data += f"Output:\n{data['output']}\n"
                diagnostic_data += "=" * 60 + "\n\n"
            
            # Create advisory agent  
            advisory_agent = Agent(
                role="System Performance Advisor",
                goal="Analyze REAL system diagnostic output and provide actionable recommendations",
                backstory="""You are a senior systems performance consultant with experience 
                in both Linux and macOS system optimization. You analyze real system metrics 
                and provide clear, actionable recommendations. You understand CPU load, memory 
                usage, network performance, and I/O operations.""",
                verbose=verbose,
                llm=self.llm,
                allow_delegation=False
            )
            
            # Create advisory task with real diagnostic data
            advisory_task = Task(
                description=f"""Analyze this REAL system diagnostic output and provide recommendations:

{diagnostic_data}

Provide:
1. **System Health Assessment**: Current state based on actual metrics
2. **CPU & Load Analysis**: Interpret actual load averages
3. **Memory Analysis**: Analyze real memory usage
4. **Network Analysis**: Review actual network data
5. **Process Analysis**: Identify resource-consuming processes
6. **Recommendations**: Specific actions based on real data
7. **Priority**: Rank by actual urgency

Base analysis on the actual command output provided.""",
                agent=advisory_agent,
                expected_output="""Comprehensive system analysis with:
1. Executive Summary
2. Detailed analysis of each area
3. Prioritized recommendations
4. Specific commands to run
Format with clear headers and priority indicators."""
            )
            
            # Create and execute crew
            crew = Crew(
                agents=[advisory_agent],
                tasks=[advisory_task],
                process=Process.sequential,
                verbose=verbose
            )
            
            print("🔍 Analyzing real system diagnostics...")
            result = crew.kickoff()
            
            return {
                'success': True,
                'analysis': str(result),
                'raw_commands': command_results,
                'llm_used': f"{self.llm_type} ({self.model_name})"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'llm_used': f"{self.llm_type} ({self.model_name})"
            }


def main():
    parser = argparse.ArgumentParser(description='Standalone System Diagnostics Agent')
    parser.add_argument('--openai-api-key', help='OpenAI API key')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print("🔧 Starting Standalone System Diagnostics...")
    print("📋 LLM Priority: Ollama (local) → OpenAI (cloud)")
    
    try:
        diagnostics = StandaloneSystemDiagnostics(openai_api_key=args.openai_api_key)
        result = diagnostics.run_diagnostics(verbose=args.verbose)
        
        if result['success']:
            print("\n" + "="*80)
            print("🔍 SYSTEM DIAGNOSTIC REPORT")
            print(f"🤖 Generated by: {result['llm_used']}")
            print("="*80)
            print(result['analysis'])
            print("="*80)
            
            print("\n💡 Next Steps:")
            print("• Review the recommendations above")
            print("• Execute suggested commands to optimize performance")
            print("• Monitor system performance after changes")
        else:
            print(f"❌ Error: {result['error']}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 