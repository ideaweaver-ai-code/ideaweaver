"""
Kubernetes Manager for IdeaWeaver
Handles Kubernetes deployment of Docker images to kind clusters
"""

import os
import json
import yaml
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
import click


class KubernetesManager:
    """Manages Kubernetes operations for model deployment"""
    
    def __init__(self, cluster_name: str = "ideaweaver-cluster", verbose: bool = False):
        """
        Initialize Kubernetes manager
        
        Args:
            cluster_name: Name of the kind cluster
            verbose: Enable verbose logging
        """
        self.cluster_name = cluster_name
        self.verbose = verbose
        
        # Check if required tools are available
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required tools (kind, kubectl, docker) are available"""
        tools = ['kind', 'kubectl', 'docker']
        missing = []
        
        for tool in tools:
            try:
                subprocess.run([tool, '--version'], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                missing.append(tool)
        
        if missing:
            raise RuntimeError(f"Missing required tools: {missing}. Please install them first.")
    
    def create_kind_cluster(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a kind cluster for model deployment
        
        Args:
            config: Custom kind cluster configuration
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if cluster already exists
            if self._cluster_exists():
                if self.verbose:
                    click.echo(f"ℹ️  Cluster '{self.cluster_name}' already exists")
                return True
            
            if self.verbose:
                click.echo(f"🔧 Creating kind cluster: {self.cluster_name}")
            
            # Default kind configuration
            default_config = {
                'kind': 'Cluster',
                'apiVersion': 'kind.x-k8s.io/v1alpha4',
                'nodes': [
                    {
                        'role': 'control-plane',
                        'extraPortMappings': [
                            {
                                'containerPort': 30080,
                                'hostPort': 8080,
                                'protocol': 'TCP'
                            }
                        ]
                    }
                ]
            }
            
            # Use custom config if provided
            cluster_config = config or default_config
            
            # Create temporary config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(cluster_config, f, default_flow_style=False)
                config_path = f.name
            
            try:
                # Create cluster
                cmd = ['kind', 'create', 'cluster', '--name', self.cluster_name, '--config', config_path]
                
                if self.verbose:
                    click.echo(f"🔧 Running: {' '.join(cmd)}")
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    click.echo(f"✅ Kind cluster created successfully: {self.cluster_name}")
                    if self.verbose and result.stdout:
                        click.echo(result.stdout)
                    return True
                else:
                    click.echo(f"❌ Failed to create cluster: {result.stderr}")
                    return False
            finally:
                # Clean up temp file
                os.unlink(config_path)
                
        except Exception as e:
            click.echo(f"❌ Error creating cluster: {str(e)}")
            return False
    
    def _cluster_exists(self) -> bool:
        """Check if the kind cluster exists"""
        try:
            result = subprocess.run(
                ['kind', 'get', 'clusters'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                clusters = result.stdout.strip().split('\n')
                return self.cluster_name in clusters
            return False
            
        except Exception:
            return False
    
    def delete_kind_cluster(self) -> bool:
        """Delete the kind cluster"""
        try:
            if not self._cluster_exists():
                click.echo(f"ℹ️  Cluster '{self.cluster_name}' does not exist")
                return True
            
            if self.verbose:
                click.echo(f"🗑️  Deleting kind cluster: {self.cluster_name}")
            
            result = subprocess.run(
                ['kind', 'delete', 'cluster', '--name', self.cluster_name],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                click.echo(f"✅ Cluster deleted successfully: {self.cluster_name}")
                return True
            else:
                click.echo(f"❌ Failed to delete cluster: {result.stderr}")
                return False
                
        except Exception as e:
            click.echo(f"❌ Error deleting cluster: {str(e)}")
            return False
    
    def load_docker_image(self, image_name: str) -> bool:
        """
        Load Docker image into kind cluster
        
        Args:
            image_name: Name of the Docker image to load
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.verbose:
                click.echo(f"📦 Loading Docker image into cluster: {image_name}")
            
            # Load image into kind cluster
            cmd = ['kind', 'load', 'docker-image', image_name, '--name', self.cluster_name]
            
            if self.verbose:
                click.echo(f"🔧 Running: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                click.echo(f"✅ Image loaded into cluster: {image_name}")
                if self.verbose and result.stdout:
                    click.echo(result.stdout)
                return True
            else:
                click.echo(f"❌ Failed to load image: {result.stderr}")
                return False
                
        except Exception as e:
            click.echo(f"❌ Error loading image: {str(e)}")
            return False
    
    def deploy_model(self, 
                    image_name: str,
                    deployment_name: str,
                    namespace: str = "default",
                    replicas: int = 1,
                    port: int = 8000,
                    service_type: str = "NodePort",
                    node_port: int = 30080,
                    resource_limits: Optional[Dict[str, str]] = None) -> bool:
        """
        Deploy model to Kubernetes cluster
        
        Args:
            image_name: Docker image name
            deployment_name: Kubernetes deployment name
            namespace: Kubernetes namespace
            replicas: Number of replicas
            port: Container port
            service_type: Kubernetes service type (NodePort, ClusterIP, LoadBalancer)
            node_port: NodePort port number (for NodePort service type)
            resource_limits: Resource limits for the container
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.verbose:
                click.echo(f"🚀 Deploying model to Kubernetes: {deployment_name}")
                click.echo(f"   Image: {image_name}")
                click.echo(f"   Namespace: {namespace}")
                click.echo(f"   Replicas: {replicas}")
            
            # Set kubectl context to kind cluster
            self._set_kubectl_context()
            
            # Create namespace if it doesn't exist
            self._create_namespace(namespace)
            
            # Generate Kubernetes manifests
            deployment_manifest = self._generate_deployment_manifest(
                image_name=image_name,
                deployment_name=deployment_name,
                namespace=namespace,
                replicas=replicas,
                port=port,
                resource_limits=resource_limits
            )
            
            service_manifest = self._generate_service_manifest(
                deployment_name=deployment_name,
                namespace=namespace,
                port=port,
                service_type=service_type,
                node_port=node_port if service_type == "NodePort" else None
            )
            
            # Apply manifests
            success = True
            success &= self._apply_manifest(deployment_manifest, "deployment")
            success &= self._apply_manifest(service_manifest, "service")
            
            if success:
                click.echo(f"✅ Model deployed successfully: {deployment_name}")
                self._show_deployment_info(deployment_name, namespace, service_type, node_port)
                return True
            else:
                click.echo(f"❌ Failed to deploy model")
                return False
                
        except Exception as e:
            click.echo(f"❌ Error deploying model: {str(e)}")
            return False
    
    def _set_kubectl_context(self):
        """Set kubectl context to the kind cluster"""
        try:
            context_name = f"kind-{self.cluster_name}"
            subprocess.run(
                ['kubectl', 'config', 'use-context', context_name],
                capture_output=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to set kubectl context: {e}")
    
    def _create_namespace(self, namespace: str):
        """Create namespace if it doesn't exist"""
        if namespace == "default":
            return
        
        try:
            subprocess.run(
                ['kubectl', 'create', 'namespace', namespace],
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError:
            # Namespace might already exist, which is fine
            pass
    
    def _generate_deployment_manifest(self, 
                                    image_name: str,
                                    deployment_name: str,
                                    namespace: str,
                                    replicas: int,
                                    port: int,
                                    resource_limits: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Generate Kubernetes deployment manifest"""
        
        container_spec = {
            'name': deployment_name,
            'image': image_name,
            'imagePullPolicy': 'Never',  # Use local image loaded into kind
            'ports': [
                {
                    'containerPort': port,
                    'name': 'http'
                }
            ],
            'livenessProbe': {
                'httpGet': {
                    'path': '/health',
                    'port': port
                },
                'initialDelaySeconds': 30,
                'periodSeconds': 10
            },
            'readinessProbe': {
                'httpGet': {
                    'path': '/health',
                    'port': port
                },
                'initialDelaySeconds': 5,
                'periodSeconds': 5
            }
        }
        
        # Add resource limits if specified
        if resource_limits:
            container_spec['resources'] = {
                'limits': resource_limits,
                'requests': {k: v for k, v in resource_limits.items()}
            }
        
        manifest = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': deployment_name,
                'namespace': namespace,
                'labels': {
                    'app': deployment_name,
                    'managed-by': 'ideaweaver'
                }
            },
            'spec': {
                'replicas': replicas,
                'selector': {
                    'matchLabels': {
                        'app': deployment_name
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': deployment_name
                        }
                    },
                    'spec': {
                        'containers': [container_spec]
                    }
                }
            }
        }
        
        return manifest
    
    def _generate_service_manifest(self, 
                                 deployment_name: str,
                                 namespace: str,
                                 port: int,
                                 service_type: str,
                                 node_port: Optional[int] = None) -> Dict[str, Any]:
        """Generate Kubernetes service manifest"""
        
        service_spec = {
            'type': service_type,
            'selector': {
                'app': deployment_name
            },
            'ports': [
                {
                    'port': port,
                    'targetPort': port,
                    'protocol': 'TCP',
                    'name': 'http'
                }
            ]
        }
        
        # Add nodePort for NodePort service type
        if service_type == "NodePort" and node_port:
            service_spec['ports'][0]['nodePort'] = node_port
        
        manifest = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f"{deployment_name}-service",
                'namespace': namespace,
                'labels': {
                    'app': deployment_name,
                    'managed-by': 'ideaweaver'
                }
            },
            'spec': service_spec
        }
        
        return manifest
    
    def _apply_manifest(self, manifest: Dict[str, Any], resource_type: str) -> bool:
        """Apply Kubernetes manifest"""
        try:
            # Create temporary manifest file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(manifest, f, default_flow_style=False)
                manifest_path = f.name
            
            try:
                # Apply manifest
                cmd = ['kubectl', 'apply', '-f', manifest_path]
                
                if self.verbose:
                    click.echo(f"🔧 Applying {resource_type} manifest")
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    if self.verbose:
                        click.echo(f"✅ {resource_type.capitalize()} applied successfully")
                    return True
                else:
                    click.echo(f"❌ Failed to apply {resource_type}: {result.stderr}")
                    return False
            finally:
                # Clean up temp file
                os.unlink(manifest_path)
                
        except Exception as e:
            click.echo(f"❌ Error applying {resource_type} manifest: {str(e)}")
            return False
    
    def _show_deployment_info(self, deployment_name: str, namespace: str, service_type: str, node_port: int):
        """Show deployment information"""
        try:
            click.echo("\n📋 Deployment Information:")
            click.echo(f"   Deployment: {deployment_name}")
            click.echo(f"   Namespace: {namespace}")
            click.echo(f"   Service Type: {service_type}")
            
            if service_type == "NodePort":
                click.echo(f"   Access URL: http://localhost:{node_port}")
                click.echo(f"   Health Check: http://localhost:{node_port}/health")
                click.echo(f"   API Docs: http://localhost:{node_port}/docs")
            
            click.echo("\n🔍 Useful Commands:")
            click.echo(f"   Check pods: kubectl get pods -n {namespace}")
            click.echo(f"   Check service: kubectl get svc -n {namespace}")
            click.echo(f"   View logs: kubectl logs -l app={deployment_name} -n {namespace}")
            click.echo(f"   Delete deployment: kubectl delete deployment {deployment_name} -n {namespace}")
            
        except Exception:
            pass  # Don't fail deployment if info display fails
    
    def undeploy_model(self, deployment_name: str, namespace: str = "default") -> bool:
        """
        Remove model deployment from Kubernetes
        
        Args:
            deployment_name: Name of the deployment to remove
            namespace: Kubernetes namespace
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.verbose:
                click.echo(f"🗑️  Removing deployment: {deployment_name}")
            
            # Set kubectl context
            self._set_kubectl_context()
            
            # Delete deployment
            result1 = subprocess.run(
                ['kubectl', 'delete', 'deployment', deployment_name, '-n', namespace],
                capture_output=True,
                text=True
            )
            
            # Delete service
            service_name = f"{deployment_name}-service"
            result2 = subprocess.run(
                ['kubectl', 'delete', 'service', service_name, '-n', namespace],
                capture_output=True,
                text=True
            )
            
            if result1.returncode == 0 and result2.returncode == 0:
                click.echo(f"✅ Deployment removed successfully: {deployment_name}")
                return True
            else:
                click.echo(f"❌ Failed to remove deployment")
                if result1.returncode != 0:
                    click.echo(f"Deployment error: {result1.stderr}")
                if result2.returncode != 0:
                    click.echo(f"Service error: {result2.stderr}")
                return False
                
        except Exception as e:
            click.echo(f"❌ Error removing deployment: {str(e)}")
            return False
    
    def list_deployments(self, namespace: str = "default") -> List[Dict[str, str]]:
        """List IdeaWeaver deployments in the cluster"""
        try:
            # Set kubectl context
            self._set_kubectl_context()
            
            # Get deployments with IdeaWeaver label
            result = subprocess.run(
                ['kubectl', 'get', 'deployments', '-n', namespace, 
                 '-l', 'managed-by=ideaweaver', '-o', 'json'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                deployments = []
                
                for item in data.get('items', []):
                    deployments.append({
                        'name': item['metadata']['name'],
                        'namespace': item['metadata']['namespace'],
                        'replicas': item['spec']['replicas'],
                        'ready_replicas': item.get('status', {}).get('readyReplicas', 0),
                        'created': item['metadata']['creationTimestamp']
                    })
                
                return deployments
            else:
                return []
                
        except Exception as e:
            if self.verbose:
                click.echo(f"Error listing deployments: {str(e)}")
            return []
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about the kind cluster"""
        try:
            info = {
                'cluster_name': self.cluster_name,
                'exists': self._cluster_exists(),
                'context': f"kind-{self.cluster_name}"
            }
            
            if info['exists']:
                # Get cluster info
                result = subprocess.run(
                    ['kubectl', 'cluster-info', '--context', info['context']],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    info['cluster_info'] = result.stdout
                
                # Get nodes
                result = subprocess.run(
                    ['kubectl', 'get', 'nodes', '--context', info['context'], '-o', 'json'],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    nodes_data = json.loads(result.stdout)
                    info['nodes'] = len(nodes_data.get('items', []))
            
            return info
            
        except Exception as e:
            return {
                'cluster_name': self.cluster_name,
                'exists': False,
                'error': str(e)
            }


def deploy_model_to_kind(image_name: str, 
                        deployment_name: str,
                        cluster_name: str = "ideaweaver-cluster",
                        **kwargs) -> bool:
    """
    Convenience function to deploy model to kind cluster
    
    Args:
        image_name: Docker image name
        deployment_name: Kubernetes deployment name
        cluster_name: Kind cluster name
        **kwargs: Additional arguments for KubernetesManager.deploy_model()
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        manager = KubernetesManager(cluster_name, verbose=kwargs.get('verbose', False))
        
        # Create cluster if it doesn't exist
        if not manager._cluster_exists():
            manager.create_kind_cluster()
        
        # Load Docker image into cluster
        manager.load_docker_image(image_name)
        
        # Deploy model
        return manager.deploy_model(image_name, deployment_name, **kwargs)
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")
        return False 