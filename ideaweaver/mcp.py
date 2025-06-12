"""
IdeaWeaver Model Context Protocol (MCP) Integration

This module provides seamless integration with popular MCP servers, allowing AI models
to interact with external services like GitHub, Slack, AWS, and more through a 
standardized protocol.
"""

import json
import os
import sys
import subprocess
import asyncio
import logging
import tempfile
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import aiohttp

try:
    from mcp import ClientSession, StdioServerParameters, stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # Using logging instead of print for better control
    import warnings
    warnings.warn(
        "MCP (Model Context Protocol) not available. "
        "Run 'pip install mcp' to enable MCP server integrations. "
        "MCP functionality will be disabled until installed.", 
        ImportWarning
    )

@dataclass
class MCPServerConfig:
    """Configuration for an MCP server"""
    name: str
    display_name: str
    description: str
    command: str
    args: List[str]
    env: Dict[str, str] = None
    transport: str = "stdio"
    enabled: bool = True
    requires_auth: bool = False
    auth_instructions: str = ""
    
    def __post_init__(self):
        if self.env is None:
            self.env = {}

@dataclass
class MCPConnection:
    """Information about an active MCP connection"""
    server_name: str
    session: Optional[Any] = None
    capabilities: Dict[str, Any] = None
    status: str = "disconnected"
    last_error: str = ""
    server_params: Optional[Any] = None

class MCPManager:
    """Manages Model Context Protocol integrations for IdeaWeaver"""
    
    def __init__(self, config_dir: str = None, verbose: bool = False):
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        
        # Configuration directory
        if config_dir is None:
            config_dir = os.path.expanduser("~/.ideaweaver/mcp")
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Pre-configured servers
        self.builtin_servers = self._get_builtin_servers()
        
        # Active connections
        self.connections: Dict[str, MCPConnection] = {}
        
        # Load user configuration
        self.user_config = self._load_user_config()
    
    def _get_builtin_servers(self) -> Dict[str, MCPServerConfig]:
        """Get pre-configured popular MCP servers"""
        servers = {}
        
        # GitHub MCP Server
        servers["github"] = MCPServerConfig(
            name="github",
            display_name="GitHub",
            description="Access GitHub repositories, issues, PRs, and more",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            requires_auth=True,
            auth_instructions=(
                "Set GITHUB_PERSONAL_ACCESS_TOKEN environment variable with your GitHub token. "
                "Create one at: https://github.com/settings/tokens"
            ),
            env={}
        )
        
        # Terraform MCP Server
        servers["terraform"] = MCPServerConfig(
            name="terraform",
            display_name="Terraform Registry",
            description="Terraform provider and module documentation, resource search and exploration",
            command="docker",
            args=["run", "-i", "--rm", "hashicorp/terraform-mcp-server"],
            requires_auth=False,
            auth_instructions="No authentication required - accesses public Terraform Registry"
        )
        
        # AWS CloudFormation MCP Server
        servers["awslabs.cfn-mcp-server"] = MCPServerConfig(
            name="awslabs.cfn-mcp-server",
            display_name="AWS CloudFormation",
            description="AWS CloudFormation resource management and operations - create, read, update, delete AWS resources",
            command="uvx",
            args=["awslabs.cfn-mcp-server@latest"],
            requires_auth=True,
            auth_instructions=(
                "Configure AWS credentials using AWS CLI or environment variables. "
                "Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_DEFAULT_REGION"
            ),
            env={
                "AWS_PROFILE": "default",
                "AWS_REGION": "us-east-1",
                "FASTMCP_LOG_LEVEL": "ERROR"
            }
        )
        
        return servers
    
    def _load_user_config(self) -> Dict[str, Any]:
        """Load user configuration from file"""
        config_file = self.config_dir / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load user config: {e}")
        
        return {
            "enabled_servers": [],
            "custom_servers": {},
            "default_settings": {
                "auto_connect": True,
                "timeout": 30
            }
        }
    
    def _save_user_config(self):
        """Save user configuration to file"""
        config_file = self.config_dir / "config.json"
        try:
            with open(config_file, 'w') as f:
                json.dump(self.user_config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save user config: {e}")
    
    def list_available_servers(self) -> List[Dict[str, Any]]:
        """List all available MCP servers (builtin + custom)"""
        servers = []
        
        # Add builtin servers
        for name, config in self.builtin_servers.items():
            server_info = asdict(config)
            server_info["type"] = "builtin"
            server_info["status"] = "connected" if name in self.connections else "available"
            servers.append(server_info)
        
        # Add custom servers
        for name, config in self.user_config.get("custom_servers", {}).items():
            server_info = config.copy()
            server_info["type"] = "custom"
            server_info["status"] = "connected" if name in self.connections else "available"
            servers.append(server_info)
        
        return servers
    
    def get_server_info(self, server_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific server"""
        if server_name in self.builtin_servers:
            config = self.builtin_servers[server_name]
            info = asdict(config)
            info["type"] = "builtin"
        elif server_name in self.user_config.get("custom_servers", {}):
            info = self.user_config["custom_servers"][server_name].copy()
            info["type"] = "custom"
        else:
            return None
        
        # Add connection status
        if server_name in self.connections:
            connection = self.connections[server_name]
            info["status"] = connection.status
            info["capabilities"] = connection.capabilities
            info["last_error"] = connection.last_error
        else:
            info["status"] = "disconnected"
        
        return info
    
    def enable_server(self, server_name: str, **kwargs):
        """Enable and configure a server"""
        if server_name not in self.builtin_servers and server_name not in self.user_config.get("custom_servers", {}):
            raise ValueError(f"Unknown server: {server_name}")
        
        # Add to enabled servers list
        if server_name not in self.user_config["enabled_servers"]:
            self.user_config["enabled_servers"].append(server_name)
        
        # Store any additional configuration
        if kwargs:
            if "server_configs" not in self.user_config:
                self.user_config["server_configs"] = {}
            self.user_config["server_configs"][server_name] = kwargs
        
        self._save_user_config()
        
        if self.verbose:
            print(f"✅ Enabled MCP server: {server_name}")
    
    def disable_server(self, server_name: str):
        """Disable a server"""
        if server_name in self.user_config["enabled_servers"]:
            self.user_config["enabled_servers"].remove(server_name)
            self._save_user_config()
        
        # Disconnect if connected
        if server_name in self.connections:
            self.disconnect_server(server_name)
        
        if self.verbose:
            print(f"❌ Disabled MCP server: {server_name}")
    
    def add_custom_server(self, config: MCPServerConfig):
        """Add a custom MCP server configuration"""
        if "custom_servers" not in self.user_config:
            self.user_config["custom_servers"] = {}
        
        self.user_config["custom_servers"][config.name] = asdict(config)
        self._save_user_config()
        
        if self.verbose:
            print(f"➕ Added custom MCP server: {config.name}")
    
    async def connect_server(self, server_name: str, **kwargs) -> bool:
        """Test connection to an MCP server"""
        if not MCP_AVAILABLE:
            raise RuntimeError("MCP not available. Install with: pip install mcp")
        
        # Get server configuration
        if server_name in self.builtin_servers:
            config = self.builtin_servers[server_name]
        elif server_name in self.user_config.get("custom_servers", {}):
            config_dict = self.user_config["custom_servers"][server_name]
            config = MCPServerConfig(**config_dict)
        else:
            raise ValueError(f"Unknown server: {server_name}")
        
        # Apply user-specific configuration
        user_config = self.user_config.get("server_configs", {}).get(server_name, {})
        kwargs.update(user_config)
        
        try:
            # Prepare environment
            env = config.env.copy()
            env.update(kwargs.get("env", {}))
            
            # Prepare command arguments
            args = config.args.copy()
            if "args" in kwargs:
                args.extend(kwargs["args"])
            
            # Set up server parameters
            server_params = StdioServerParameters(
                command=config.command,
                args=args,
                env=env if env else None
            )
            
            if self.verbose:
                print(f"🔗 Connecting to MCP server: {server_name}")
                print(f"   Command: {config.command}")
                print(f"   Args: {args}")
            
            # Test connection
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # Get server capabilities
                    try:
                        prompts = await session.list_prompts()
                        resources = await session.list_resources()
                        tools = await session.list_tools()
                        
                        capabilities = {
                            "prompts": [p.name for p in prompts.prompts] if prompts else [],
                            "resources": [r.uri for r in resources.resources] if resources else [],
                            "tools": [t.name for t in tools.tools] if tools else []
                        }
                    except Exception as e:
                        capabilities = {"error": str(e)}
                    
                    # Store connection info (without session)
                    connection = MCPConnection(
                        server_name=server_name,
                        session=None,  # Don't store session, create on-demand
                        capabilities=capabilities,
                        status="connected"
                    )
                    self.connections[server_name] = connection
                    
                    # Store server params for later use
                    connection.server_params = server_params
                    
                    if self.verbose:
                        print(f"✅ Successfully connected to {server_name}")
                        if capabilities.get("tools"):
                            print(f"   Available tools: {', '.join(capabilities['tools'])}")
                        if capabilities.get("resources"):
                            print(f"   Available resources: {len(capabilities['resources'])}")
                    
                    return True
        
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Failed to connect to {server_name}: {error_msg}")
            
            # Store failed connection info
            connection = MCPConnection(
                server_name=server_name,
                status="error",
                last_error=error_msg
            )
            self.connections[server_name] = connection
            
            if self.verbose:
                print(f"❌ Failed to connect to {server_name}: {error_msg}")
            
            return False
    
    def disconnect_server(self, server_name: str):
        """Disconnect from an MCP server"""
        if server_name in self.connections:
            connection = self.connections[server_name]
            if connection.session:
                # Session cleanup is handled by async context manager
                pass
            
            del self.connections[server_name]
            
            if self.verbose:
                print(f"🔌 Disconnected from MCP server: {server_name}")
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on an MCP server"""
        if server_name not in self.connections:
            raise ValueError(f"Not connected to server: {server_name}")
        
        connection = self.connections[server_name]
        if connection.status != "connected":
            raise ValueError(f"Server {server_name} is not properly connected")
        
        # Get stored server params
        if not hasattr(connection, 'server_params'):
            raise ValueError(f"No server parameters stored for {server_name}")
        
        try:
            # Create fresh connection for each tool call
            async with stdio_client(connection.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
                    return {
                        "success": True,
                        "result": result.content,
                        "server": server_name,
                        "tool": tool_name
                    }
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Tool call failed - {server_name}.{tool_name}: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "server": server_name,
                "tool": tool_name
            }
    
    async def read_resource(self, server_name: str, uri: str) -> Dict[str, Any]:
        """Read a resource from an MCP server"""
        if server_name not in self.connections:
            raise ValueError(f"Not connected to server: {server_name}")
        
        connection = self.connections[server_name]
        if connection.status != "connected" or not connection.session:
            raise ValueError(f"Server {server_name} is not properly connected")
        
        try:
            result = await connection.session.read_resource(uri)
            return {
                "success": True,
                "content": result.contents,
                "server": server_name,
                "uri": uri
            }
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Resource read failed - {server_name} / {uri}: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "server": server_name,
                "uri": uri
            }
    
    def get_connection_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all connections"""
        status = {}
        for server_name, connection in self.connections.items():
            status[server_name] = {
                "status": connection.status,
                "capabilities": connection.capabilities,
                "last_error": connection.last_error
            }
        return status
    
    def generate_claude_desktop_config(self, output_file: str = None) -> Dict[str, Any]:
        """Generate Claude Desktop configuration for enabled servers"""
        claude_config = {"mcpServers": {}}
        
        for server_name in self.user_config.get("enabled_servers", []):
            if server_name in self.builtin_servers:
                config = self.builtin_servers[server_name]
                user_config = self.user_config.get("server_configs", {}).get(server_name, {})
                
                server_config = {
                    "command": config.command,
                    "args": config.args.copy()
                }
                
                # Add user-specific args
                if "args" in user_config:
                    server_config["args"].extend(user_config["args"])
                
                # Add environment variables
                env = config.env.copy()
                env.update(user_config.get("env", {}))
                if env:
                    server_config["env"] = env
                
                claude_config["mcpServers"][server_name] = server_config
        
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    json.dump(claude_config, f, indent=2)
                if self.verbose:
                    print(f"📝 Claude Desktop config saved to: {output_file}")
            except Exception as e:
                self.logger.error(f"Failed to save Claude config: {e}")
        
        return claude_config
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check if required dependencies are available"""
        deps = {}
        
        # Check MCP Python package
        deps["mcp"] = MCP_AVAILABLE
        
        # Check Node.js and npm (for most builtin servers)
        try:
            result = subprocess.run(["node", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            deps["node"] = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            deps["node"] = False
        
        try:
            result = subprocess.run(["npm", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            deps["npm"] = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            deps["npm"] = False
        
        # Check npx specifically
        try:
            result = subprocess.run(["npx", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            deps["npx"] = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            deps["npx"] = False
        
        return deps
    
    def setup_authentication(self, server_name: str, **auth_params) -> bool:
        """Help set up authentication for a server"""
        if server_name not in self.builtin_servers:
            if self.verbose:
                print(f"❌ Unknown builtin server: {server_name}")
            return False
        
        config = self.builtin_servers[server_name]
        
        if not config.requires_auth:
            if self.verbose:
                print(f"✅ Server {server_name} does not require authentication")
            return True
        
        if self.verbose:
            print(f"🔐 Setting up authentication for {server_name}")
            print(f"   Instructions: {config.auth_instructions}")
        
        # Store auth configuration
        if "server_configs" not in self.user_config:
            self.user_config["server_configs"] = {}
        
        if server_name not in self.user_config["server_configs"]:
            self.user_config["server_configs"][server_name] = {}
        
        # Store provided auth parameters
        if auth_params:
            self.user_config["server_configs"][server_name].update(auth_params)
            self._save_user_config()
            
            if self.verbose:
                print("✅ Authentication configuration saved")
        
        return True

# Utility functions for integration with IdeaWeaver CLI

def get_mcp_manager(verbose: bool = False) -> MCPManager:
    """Get or create global MCP manager instance"""
    if not hasattr(get_mcp_manager, '_instance'):
        get_mcp_manager._instance = MCPManager(verbose=verbose)
    return get_mcp_manager._instance

async def execute_mcp_tool(server_name: str, tool_name: str, arguments: Dict[str, Any], 
                          verbose: bool = False) -> Dict[str, Any]:
    """High-level function to execute an MCP tool"""
    manager = get_mcp_manager(verbose=verbose)
    
    # Auto-connect if not connected
    if server_name not in manager.connections:
        success = await manager.connect_server(server_name)
        if not success:
            return {
                "success": False,
                "error": f"Failed to connect to server: {server_name}"
            }
    
    return await manager.call_tool(server_name, tool_name, arguments)

def format_mcp_result(result: Dict[str, Any], show_raw: bool = False) -> str:
    """Format MCP result for display"""
    if not result.get("success", False):
        return f"❌ Error: {result.get('error', 'Unknown error')}"
    
    if show_raw:
        return f"Raw result: {json.dumps(result, indent=2)}"
    
    server_name = result.get('server', 'unknown')
    tool_name = result.get('tool', 'unknown')
    content = result.get("result", result.get("content", ""))
    
    # Handle GitHub-specific formatting
    if server_name == "github":
        return _format_github_result(tool_name, content)
    
    # Handle AWS-specific formatting
    if server_name.startswith("aws"):
        return _format_aws_result(server_name, tool_name, content)
    
    # Handle Terraform-specific formatting
    if server_name == "terraform":
        return _format_terraform_result(tool_name, content)
    
    # Handle other server responses
    output = f"✅ Success from {server_name} server\n"
    
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and "text" in item:
                # Try to parse as JSON for better formatting
                try:
                    data = json.loads(item["text"])
                    output += _format_json_data(data) + "\n"
                except (json.JSONDecodeError, TypeError):
                    # Handle as formatted text content
                    output += _format_text_content(item["text"]) + "\n"
            else:
                output += str(item) + "\n"
    else:
        output += str(content) + "\n"
    
    return output.strip()

def _format_github_result(tool_name: str, content: Any) -> str:
    """Format GitHub-specific API responses"""
    if not content:
        return "✅ GitHub API call completed (no content returned)"
    
    # Extract text content from MCP response format
    data = None
    
    # Handle list of content items (MCP format)
    if isinstance(content, list) and len(content) > 0:
        content_item = content[0]
        
        # Check if it's an object with 'text' attribute (common MCP format)
        if hasattr(content_item, 'text'):
            try:
                data = json.loads(content_item.text)
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
        
        # Check if it's a dict with 'text' key
        elif isinstance(content_item, dict) and "text" in content_item:
            try:
                data = json.loads(content_item["text"])
            except (json.JSONDecodeError, TypeError):
                pass
        
        # If no text field, use the content item directly
        else:
            data = content_item
    
    # Handle direct content
    else:
        try:
            if isinstance(content, str):
                data = json.loads(content)
            else:
                data = content
        except (json.JSONDecodeError, TypeError):
            pass
    
    # If we still don't have data, show debug info and return raw
    if not data:
        return f"✅ GitHub {tool_name} completed\nRaw content: {str(content)[:500]}..."
    
    # Format based on tool type
    if tool_name == "search_repositories":
        return _format_repository_search(data)
    elif tool_name == "get_file_contents":
        return _format_file_contents(data)
    elif tool_name == "list_issues":
        return _format_issues_list(data)
    elif tool_name == "list_pull_requests":
        return _format_pull_requests_list(data)
    elif tool_name in ["create_issue", "create_pull_request"]:
        return _format_created_item(data, tool_name)
    else:
        # Fallback to generic formatting
        return f"✅ GitHub {tool_name} result:\n{_format_json_data(data)}"

def _format_repository_search(data: Dict[str, Any]) -> str:
    """Format GitHub repository search results"""
    if not isinstance(data, dict) or "items" not in data:
        return f"✅ GitHub repository search completed:\n{_format_json_data(data)}"
    
    total_count = data.get("total_count", 0)
    items = data.get("items", [])
    
    output = f"🔍 **GitHub Repository Search Results**\n"
    output += f"📊 Found {total_count} repositories total, showing top {len(items)}:\n\n"
    
    for i, repo in enumerate(items, 1):
        name = repo.get("full_name", repo.get("name", "Unknown"))
        description = repo.get("description") or "No description"
        
        # Get available metrics (search API doesn't return stars/forks)
        stars = repo.get("stargazers_count")
        forks = repo.get("forks_count") 
        language = repo.get("language")
        updated = repo.get("updated_at", "").split("T")[0] if repo.get("updated_at") else "N/A"
        
        # Check if this repo has detailed info (some APIs return it, some don't)
        has_detailed_info = stars is not None and forks is not None
        
        output += f"**{i}. {name}**\n"
        output += f"   📝 {description}\n"
        
        if has_detailed_info:
            output += f"   🌟 Stars: {stars} | 🍴 Forks: {forks}"
            if language:
                output += f" | 💻 Language: {language}"
            output += "\n"
        else:
            # Search API doesn't return detailed metrics
            output += f"   ℹ️  Repository found (detailed metrics not available in search results)\n"
        
        output += f"   📅 Last updated: {updated}\n"
        output += f"   🔗 {repo.get('html_url', 'N/A')}\n\n"
    
    if not items or not any(repo.get("stargazers_count") is not None for repo in items):
        output += "💡 **Tip**: Use a specific repository lookup tool for detailed metrics like stars and forks.\n"
    
    return output.strip()

def _format_file_contents(data: Dict[str, Any]) -> str:
    """Format GitHub file contents result"""
    if isinstance(data, dict) and "content" in data:
        filename = data.get("name", "file")
        size = data.get("size", 0)
        output = f"📄 **File: {filename}** ({size} bytes)\n"
        output += f"```\n{data['content']}\n```"
        return output
    return f"✅ File contents:\n{_format_json_data(data)}"

def _format_issues_list(data: Any) -> str:
    """Format GitHub issues list"""
    if not isinstance(data, list):
        return f"✅ GitHub issues:\n{_format_json_data(data)}"
    
    output = f"🐛 **GitHub Issues** ({len(data)} total):\n\n"
    
    for i, issue in enumerate(data, 1):
        title = issue.get("title", "No title")
        number = issue.get("number", "N/A")
        state = issue.get("state", "unknown")
        user = issue.get("user", {}).get("login", "unknown")
        created = issue.get("created_at", "").split("T")[0] if issue.get("created_at") else "N/A"
        
        state_icon = "🟢" if state == "open" else "🔴"
        
        output += f"**{i}. #{number}: {title}**\n"
        output += f"   {state_icon} {state.title()} | 👤 {user} | 📅 {created}\n"
        output += f"   🔗 {issue.get('html_url', 'N/A')}\n\n"
    
    return output.strip()

def _format_pull_requests_list(data: Any) -> str:
    """Format GitHub pull requests list"""
    if not isinstance(data, list):
        return f"✅ GitHub pull requests:\n{_format_json_data(data)}"
    
    if len(data) == 0:
        return "📭 **No Pull Requests Found**\nThis repository has no open or recent pull requests."
    
    output = f"🔀 **GitHub Pull Requests** ({len(data)} total):\n\n"
    
    for i, pr in enumerate(data, 1):
        title = pr.get("title", "No title")
        number = pr.get("number", "N/A")
        state = pr.get("state", "unknown")
        user = pr.get("user", {}).get("login", "unknown")
        created = pr.get("created_at", "").split("T")[0] if pr.get("created_at") else "N/A"
        
        # Pull request specific info
        head_branch = pr.get("head", {}).get("ref", "unknown")
        base_branch = pr.get("base", {}).get("ref", "unknown")
        draft = pr.get("draft", False)
        
        # State icons
        if state == "open":
            state_icon = "🟢"
            if draft:
                state_icon = "📝"  # Draft PR
        elif state == "closed":
            # Check if it was merged
            merged = pr.get("merged", False)
            state_icon = "🟣" if merged else "🔴"
        else:
            state_icon = "⚪"
        
        output += f"**{i}. #{number}: {title}**\n"
        output += f"   {state_icon} {state.title()}"
        if draft:
            output += " (Draft)"
        output += f" | 👤 {user} | 📅 {created}\n"
        output += f"   🌿 {head_branch} → {base_branch}\n"
        output += f"   🔗 {pr.get('html_url', 'N/A')}\n\n"
    
    return output.strip()

def _format_created_item(data: Dict[str, Any], tool_name: str) -> str:
    """Format newly created GitHub items (issues, PRs, etc.)"""
    item_type = "Issue" if "issue" in tool_name else "Pull Request"
    
    if isinstance(data, dict):
        title = data.get("title", "Untitled")
        number = data.get("number", "N/A")
        url = data.get("html_url", "N/A")
        
        output = f"✅ **{item_type} Created Successfully!**\n"
        output += f"📋 #{number}: {title}\n"
        output += f"🔗 {url}"
        return output
    
    return f"✅ {item_type} created:\n{_format_json_data(data)}"

def _format_json_data(data: Any) -> str:
    """Generic JSON data formatter with better structure"""
    if isinstance(data, dict):
        if len(data) <= 3:
            # Small objects, format inline
            items = [f"{k}: {v}" for k, v in data.items()]
            return " | ".join(items)
        else:
            # Larger objects, format as list
            output = ""
            for k, v in data.items():
                if isinstance(v, (dict, list)) and len(str(v)) > 50:
                    output += f"• {k}: [complex data]\n"
                else:
                    output += f"• {k}: {v}\n"
            return output.strip()
    elif isinstance(data, list):
        if len(data) <= 5:
            return ", ".join(str(item) for item in data)
        else:
            return f"[{len(data)} items: {', '.join(str(item) for item in data[:3])}, ...]"
    else:
        return str(data)

def _format_aws_result(server_name: str, tool_name: str, content: Any) -> str:
    """Format AWS-specific MCP responses"""
    if not content:
        return f"✅ AWS {tool_name} completed (no content returned)"
    
    # Extract text content from MCP response format
    text_content = None
    
    # Handle list of content items (MCP format)
    if isinstance(content, list) and len(content) > 0:
        content_item = content[0]
        
        # Check if it's an object with 'text' attribute (common MCP format)
        if hasattr(content_item, 'text'):
            text_content = content_item.text
        # Check if it's a dict with 'text' key
        elif isinstance(content_item, dict) and "text" in content_item:
            text_content = content_item["text"]
        else:
            text_content = str(content_item)
    else:
        text_content = str(content)
    
    # Format based on server type
    if server_name == "aws-documentation":
        return _format_aws_documentation_result(tool_name, text_content)
    elif server_name == "aws-cdk":
        return _format_aws_cdk_result(tool_name, text_content)
    elif server_name == "aws-core":
        return _format_aws_core_result(tool_name, text_content)
    elif server_name == "aws-kb":
        return _format_aws_kb_result(tool_name, text_content)
    else:
        # Generic AWS formatting
        return f"✅ AWS {tool_name} result:\n{_format_text_content(text_content)}"

def _format_aws_documentation_result(tool_name: str, content: str) -> str:
    """Format AWS Documentation server results"""
    if tool_name == "search_documentation":
        return f"📚 **AWS Documentation Search Results**\n\n{_format_text_content(content)}"
    elif tool_name == "read_documentation":
        return f"📖 **AWS Documentation**\n\n{_format_text_content(content)}"
    else:
        return f"📚 **AWS Documentation - {tool_name}**\n\n{_format_text_content(content)}"

def _format_aws_cdk_result(tool_name: str, content: str) -> str:
    """Format AWS CDK server results"""
    if tool_name == "CDKGeneralGuidance":
        return f"🏗️ **AWS CDK Guidance**\n\n{_format_text_content(content)}"
    elif tool_name == "ExplainCDKNagRule":
        return f"🔍 **CDK Nag Rule Explanation**\n\n{_format_text_content(content)}"
    elif tool_name == "CheckCDKNagSuppressions":
        return f"✅ **CDK Nag Suppressions Check**\n\n{_format_text_content(content)}"
    elif tool_name == "GenerateCDKNagSuppression":
        return f"🛠️ **CDK Nag Suppression Generated**\n\n{_format_text_content(content)}"
    else:
        return f"🏗️ **AWS CDK - {tool_name}**\n\n{_format_text_content(content)}"

def _format_aws_core_result(tool_name: str, content: str) -> str:
    """Format AWS Core server results"""
    return f"☁️ **AWS Core - {tool_name}**\n\n{_format_text_content(content)}"

def _format_aws_kb_result(tool_name: str, content: str) -> str:
    """Format AWS Knowledge Base server results"""
    if tool_name == "query_knowledge_base":
        return f"🧠 **AWS Knowledge Base Query**\n\n{_format_text_content(content)}"
    else:
        return f"🧠 **AWS Knowledge Base - {tool_name}**\n\n{_format_text_content(content)}"

def _format_text_content(content: str) -> str:
    """Format text content with proper line breaks and markdown rendering"""
    if not content:
        return "No content"
    
    # Clean up the content
    content = content.strip()
    
    # If it's already well-formatted (has line breaks), return as is
    if '\n' in content and len(content.split('\n')) > 3:
        return content
    
    # If it's a single long line, try to add proper formatting
    # Look for common patterns that should be on new lines
    
    # Handle markdown headers
    content = content.replace('# ', '\n# ').replace('## ', '\n## ').replace('### ', '\n### ')
    
    # Handle list items
    content = content.replace('* ', '\n* ').replace('- ', '\n- ')
    content = content.replace('1. ', '\n1. ').replace('2. ', '\n2. ').replace('3. ', '\n3. ')
    
    # Handle code blocks
    content = content.replace('```', '\n```\n')
    
    # Handle numbered steps
    import re
    content = re.sub(r'(\d+\.\s)', r'\n\1', content)
    
    # Handle sentences that should be paragraphs (end with period followed by capital letter)
    content = re.sub(r'(\.\s+)([A-Z])', r'\1\n\2', content)
    
    # Clean up multiple newlines
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
    
    # Remove leading newlines
    content = content.lstrip('\n')
    
    return content

def _format_terraform_result(tool_name: str, content: Any) -> str:
    """Format Terraform-specific MCP responses"""
    if not content:
        return f"✅ Terraform {tool_name} completed (no content returned)"
    
    # Extract text content from MCP response format
    text_content = None
    
    # Handle list of content items (MCP format)
    if isinstance(content, list) and len(content) > 0:
        content_item = content[0]
        
        # Check if it's an object with 'text' attribute (common MCP format)
        if hasattr(content_item, 'text'):
            text_content = content_item.text
        # Check if it's a dict with 'text' key
        elif isinstance(content_item, dict) and "text" in content_item:
            text_content = content_item["text"]
        else:
            text_content = str(content_item)
    else:
        text_content = str(content)
    
    # Format based on tool type
    if tool_name == "searchModules":
        return _format_terraform_module_search(text_content)
    elif tool_name == "moduleDetails":
        return _format_terraform_module_details(text_content)
    elif tool_name == "resolveProviderDocID":
        return _format_terraform_provider_docs(text_content)
    elif tool_name == "getProviderDocs":
        return _format_terraform_provider_content(text_content)
    else:
        # Generic Terraform formatting
        return f"🏗️ **Terraform {tool_name} Result**\n\n{_format_terraform_text(text_content)}"

def _format_terraform_module_search(content: str) -> str:
    """Format Terraform module search results"""
    if not content:
        return "📦 **No Terraform modules found**"
    
    # Clean up the content and format properly
    formatted_content = _format_terraform_text(content)
    
    # Add header and structure
    output = "📦 **Terraform Module Search Results**\n\n"
    output += formatted_content
    
    return output

def _format_terraform_module_details(content: str) -> str:
    """Format Terraform module details"""
    if not content:
        return "📦 **No module details available**"
    
    formatted_content = _format_terraform_text(content)
    
    output = "📋 **Terraform Module Details**\n\n"
    output += formatted_content
    
    return output

def _format_terraform_provider_docs(content: str) -> str:
    """Format Terraform provider documentation list"""
    if not content:
        return "📚 **No provider documentation found**"
    
    formatted_content = _format_terraform_text(content)
    
    output = "📚 **Terraform Provider Documentation**\n\n"
    output += formatted_content
    
    return output

def _format_terraform_provider_content(content: str) -> str:
    """Format Terraform provider documentation content"""
    if not content:
        return "📖 **No documentation content available**"
    
    formatted_content = _format_terraform_text(content)
    
    output = "📖 **Terraform Provider Documentation Content**\n\n"
    output += formatted_content
    
    return output

def _format_terraform_text(content: str) -> str:
    """Format Terraform text content with proper formatting"""
    if not content:
        return "No content available"
    
    # Clean up the content
    content = content.strip()
    
    # Replace escaped newlines with actual newlines
    content = content.replace('\\n', '\n')
    
    # Replace escaped tabs with spaces
    content = content.replace('\\t', '  ')
    
    # Handle markdown table formatting
    import re
    
    # Split into lines for processing
    lines = content.split('\n')
    formatted_lines = []
    in_table = False
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        if not line:
            formatted_lines.append('')
            continue
        
        # Detect markdown tables
        if '|' in line and line.count('|') >= 2:
            in_table = True
            # Clean up table formatting
            # Remove markdown links but keep the text
            line = re.sub(r'<a name="[^"]*"></a> \[([^\]]+)\]\([^)]+\)', r'\1', line)
            line = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', line)
            
            # Split by | and clean up each cell
            cells = [cell.strip() for cell in line.split('|')]
            if len(cells) >= 3:  # Valid table row
                # Remove empty first/last cells if they exist
                if cells[0] == '':
                    cells = cells[1:]
                if cells and cells[-1] == '':
                    cells = cells[:-1]
                
                if len(cells) >= 2:
                    # Format as a simple two-column layout
                    name = cells[0].strip()
                    desc = ' '.join(cells[1:]).strip()
                    
                    # Skip table header separators
                    if name.startswith('---') or desc.startswith('---'):
                        continue
                    
                    # Skip if it's just column headers that are too generic
                    if name.lower() in ['name', 'description', 'type', 'default', 'required']:
                        formatted_lines.append(f"**{name}** | **{desc}**")
                    else:
                        formatted_lines.append(f"• **{name}**: {desc}")
                continue
        else:
            in_table = False
        
        # Handle section headers
        if line.startswith('## '):
            formatted_lines.append(f"\n🔷 **{line[3:]}**")
        elif line.startswith('### '):
            formatted_lines.append(f"\n📋 **{line[4:]}**")
        elif line.startswith('# '):
            formatted_lines.append(f"\n🏗️ **{line[2:]}**")
        
        # Handle lists
        elif line.startswith('- '):
            formatted_lines.append(f"  {line}")
        elif line.startswith('* '):
            formatted_lines.append(f"  {line}")
        elif re.match(r'^\d+\. ', line):
            formatted_lines.append(f"  {line}")
        
        # Handle key-value pairs common in Terraform
        elif line.startswith('moduleID:') or line.startswith('Name:') or line.startswith('Description:'):
            formatted_lines.append(f"**{line}**")
        elif line.startswith('Downloads:') or line.startswith('Published:') or line.startswith('Verified:'):
            formatted_lines.append(f"  📊 {line}")
        elif line.startswith('Source:') or line.startswith('Version:'):
            formatted_lines.append(f"  🔗 {line}")
        
        # Handle code blocks
        elif line.startswith('```'):
            formatted_lines.append(line)
        
        # Regular content
        else:
            # Remove excessive markdown formatting
            line = re.sub(r'<a name="[^"]*"></a> \[([^\]]+)\]\([^)]+\)', r'\1', line)
            line = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', line)
            formatted_lines.append(line)
    
    # Clean up multiple newlines
    result = '\n'.join(formatted_lines)
    result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)
    
    # Limit very long outputs for readability
    lines = result.split('\n')
    if len(lines) > 100:
        truncated_lines = lines[:100]
        truncated_lines.append("\n... (output truncated for readability)")
        truncated_lines.append(f"📊 Total: {len(lines)} lines, showing first 100")
        result = '\n'.join(truncated_lines)
    
    return result 

async def call_llm(prompt: str) -> str:
    """Call the LLM with the given prompt.
    
    Args:
        prompt: The prompt to send to the LLM
        
    Returns:
        The LLM's response
    """
    # TODO: Implement actual LLM integration
    # For now, return a simple response
    return '{"server": "awslabs.cfn-mcp-server", "tool": "list_resources", "args": {"resource_type": "AWS::S3::Bucket"}}'

async def interpret_command_with_ollama(command: str) -> Dict[str, Any]:
    """Use Ollama to interpret natural language command into MCP format.
    
    Args:
        command: Natural language command from user
        
    Returns:
        Dict containing server, tool, and arguments for MCP
    """
    prompt = f"""Convert this AWS command into MCP format:
Command: {command}

Available tools:
- list_resources: List AWS resources of a specific type
- create_resource: Create a new AWS resource
- delete_resource: Delete an AWS resource

Return the response in this JSON format:
{{
    "server": "awslabs.cfn-mcp-server",
    "tool": "tool_name",
    "args": {{
        "resource_type": "AWS::ResourceType",
        "properties": {{
            // resource specific properties
        }}
    }}
}}

Example:
Command: "list s3 buckets"
Response: {{
    "server": "awslabs.cfn-mcp-server",
    "tool": "list_resources",
    "args": {{
        "resource_type": "AWS::S3::Bucket"
    }}
}}
"""
    
    # Call Ollama API
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3:mini",
                "prompt": prompt,
                "stream": False
            }
        ) as response:
            if response.status != 200:
                raise ValueError(f"Ollama API error: {response.status}")
            
            result = await response.json()
            response_text = result.get("response", "")
            
            # Extract JSON from response
            try:
                # Find JSON in the response
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                if json_start == -1 or json_end == 0:
                    raise ValueError("No JSON found in Ollama response")
                    
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse Ollama response as JSON: {e}")

async def execute_natural_command(command: str, verbose: bool = False) -> Dict[str, Any]:
    """Execute a natural language command using AWS MCP server.
    
    Args:
        command: Natural language command from user
        verbose: Whether to show detailed output
        
    Returns:
        Dict containing the result of the command execution
    """
    # Pass the natural language command directly to MCP server
    return await execute_mcp_tool(
        "awslabs.cfn-mcp-server",
        "execute_command",
        {"command": command},
        verbose=verbose
    )

class CommandTemplate:
    """Represents a command template with variable substitution."""
    
    def __init__(self, pattern: str, server: str, tool: str, args_template: Dict[str, str]):
        self.pattern = pattern
        self.server = server
        self.tool = tool
        self.args_template = args_template
    
    def match(self, command: str) -> Optional[Dict[str, Any]]:
        """Try to match the command against this template."""
        # Convert both to lowercase for matching
        cmd = command.lower()
        pattern = self.pattern.lower()
        
        # Simple pattern matching - can be made more sophisticated
        if pattern in cmd:
            # For fixed arguments, just return them as is
            if all(not isinstance(v, str) or not v.startswith("{") for v in self.args_template.values()):
                # Special handling for S3 bucket creation
                if self.tool == "create_resource" and self.args_template.get("resource_type") == "AWS::S3::Bucket":
                    # Extract bucket name from the command
                    bucket_name = cmd[cmd.find("name") + 4:].strip()
                    args = self.args_template.copy()
                    args["desired_state"] = {"BucketName": bucket_name}
                    return {
                        "server": self.server,
                        "tool": self.tool,
                        "args": args
                    }
                return {
                    "server": self.server,
                    "tool": self.tool,
                    "args": self.args_template
                }
            
            # For templates with variables, extract them
            args = {}
            for arg_name, arg_pattern in self.args_template.items():
                if arg_pattern in cmd:
                    # Extract the value after the pattern
                    value = cmd[cmd.find(arg_pattern) + len(arg_pattern):].strip()
                    args[arg_name] = value
            
            return {
                "server": self.server,
                "tool": self.tool,
                "args": args
            }
        return None

# Define command templates
COMMAND_TEMPLATES = [
    CommandTemplate(
        pattern="list s3 buckets",
        server="awslabs.cfn-mcp-server",
        tool="list_resources",
        args_template={"resource_type": "AWS::S3::Bucket"}
    ),
    CommandTemplate(
        pattern="create new s3 bucket name",
        server="awslabs.cfn-mcp-server",
        tool="create_resource",
        args_template={
            "resource_type": "AWS::S3::Bucket",
            "desired_state": {"BucketName": ""}
        }
    ),
    CommandTemplate(
        pattern="list ec2 instances",
        server="awslabs.cfn-mcp-server",
        tool="list_resources",
        args_template={"resource_type": "AWS::EC2::Instance"}
    ),
    CommandTemplate(
        pattern="create {resource_type}",
        server="awslabs.cfn-mcp-server",
        tool="create_resource",
        args_template={"resource_type": "resource type"}
    ),
    CommandTemplate(
        pattern="search repositories {query}",
        server="github",
        tool="search_repositories",
        args_template={"query": "query"}
    ),
] 