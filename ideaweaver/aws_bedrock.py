"""
AWS Bedrock Custom Model Import Integration
Supports importing trained models to AWS Bedrock for inference
"""

import boto3
import json
import time
import os
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


class BedrockModelImporter:
    """Handles importing custom models to AWS Bedrock"""
    
    def __init__(self, region_name: str = "us-east-1"):
        """
        Initialize Bedrock client
        
        Args:
            region_name: AWS region (must be us-east-1, us-west-2, or eu-central-1)
        """
        self.region_name = region_name
        self.supported_regions = ["us-east-1", "us-west-2", "eu-central-1"]
        
        if region_name not in self.supported_regions:
            raise ValueError(f"Bedrock Custom Model Import only supported in: {self.supported_regions}")
        
        try:
            self.bedrock_client = boto3.client("bedrock", region_name=region_name)
            self.bedrock_runtime = boto3.client("bedrock-runtime", region_name=region_name)
            self.s3_client = boto3.client("s3", region_name=region_name)
        except NoCredentialsError:
            raise ValueError("AWS credentials not found. Please configure AWS CLI or set environment variables.")
    
    def validate_model_architecture(self, model_path: str) -> bool:
        """
        Validate if model architecture is supported by Bedrock
        
        Supported architectures:
        - Mistral, Mixtral
        - Flan (T5-based)
        - Llama 2, Llama 3, Llama 3.1, Llama 3.2, Llama 3.3, Mllama
        - GPTBigCode
        - Qwen2, Qwen2.5, Qwen2-VL, Qwen2.5-VL
        """
        config_path = Path(model_path) / "config.json"
        
        if not config_path.exists():
            logger.error(f"config.json not found in {model_path}")
            return False
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            model_type = config.get("model_type", "").lower()
            architectures = config.get("architectures", [])
            
            supported_types = [
                "mistral", "mixtral", "llama", "gpt_bigcode", 
                "qwen2", "t5", "flan-t5"
            ]
            
            supported_architectures = [
                "MistralForCausalLM", "MixtralForCausalLM",
                "LlamaForCausalLM", "GPTBigCodeForCausalLM",
                "Qwen2ForCausalLM", "T5ForConditionalGeneration"
            ]
            
            is_supported = (
                any(stype in model_type for stype in supported_types) or
                any(arch in architectures for arch in supported_architectures)
            )
            
            if not is_supported:
                logger.warning(f"Model architecture may not be supported: {model_type}, {architectures}")
                logger.info("Supported architectures: Mistral, Mixtral, Llama, GPTBigCode, Qwen2, T5/Flan")
            
            return True  # Allow attempt even if uncertain
            
        except Exception as e:
            logger.error(f"Error validating model architecture: {e}")
            return False
    
    def validate_model_files(self, model_path: str) -> Dict[str, bool]:
        """
        Validate required model files are present
        
        Required files:
        - model.safetensors (or model-*.safetensors)
        - config.json
        - tokenizer_config.json
        - tokenizer.json
        - tokenizer.model (for some models)
        """
        model_dir = Path(model_path)
        
        validation = {
            "safetensors": False,
            "config": False,
            "tokenizer_config": False,
            "tokenizer_json": False,
            "tokenizer_model": False
        }
        
        # Check for safetensors files
        safetensor_files = list(model_dir.glob("*.safetensors"))
        validation["safetensors"] = len(safetensor_files) > 0
        
        # Check required JSON files
        validation["config"] = (model_dir / "config.json").exists()
        validation["tokenizer_config"] = (model_dir / "tokenizer_config.json").exists()
        validation["tokenizer_json"] = (model_dir / "tokenizer.json").exists()
        
        # tokenizer.model is optional (for SentencePiece models)
        validation["tokenizer_model"] = (model_dir / "tokenizer.model").exists()
        
        return validation
    
    def upload_model_to_s3(self, model_path: str, s3_bucket: str, s3_prefix: str) -> str:
        """
        Upload model files to S3
        
        Args:
            model_path: Local path to model directory
            s3_bucket: S3 bucket name
            s3_prefix: S3 prefix/folder path
            
        Returns:
            S3 URI of uploaded model
        """
        model_dir = Path(model_path)
        
        if not model_dir.exists():
            raise ValueError(f"Model directory not found: {model_path}")
        
        # Validate model files
        validation = self.validate_model_files(model_path)
        required_files = ["safetensors", "config", "tokenizer_config", "tokenizer_json"]
        
        missing_files = [f for f in required_files if not validation[f]]
        if missing_files:
            raise ValueError(f"Missing required files: {missing_files}")
        
        logger.info(f"📤 Uploading model to S3: s3://{s3_bucket}/{s3_prefix}")
        
        try:
            # Upload all files in model directory
            uploaded_files = []
            for file_path in model_dir.iterdir():
                if file_path.is_file():
                    s3_key = f"{s3_prefix}/{file_path.name}"
                    
                    logger.info(f"  Uploading {file_path.name}...")
                    self.s3_client.upload_file(
                        str(file_path), 
                        s3_bucket, 
                        s3_key
                    )
                    uploaded_files.append(file_path.name)
            
            s3_uri = f"s3://{s3_bucket}/{s3_prefix}/"
            logger.info(f"✅ Model uploaded successfully to {s3_uri}")
            logger.info(f"📁 Uploaded files: {uploaded_files}")
            
            return s3_uri
            
        except ClientError as e:
            logger.error(f"Failed to upload to S3: {e}")
            raise
    
    def create_import_job(
        self, 
        job_name: str,
        model_name: str, 
        s3_uri: str,
        role_arn: str,
        inference_type: str = "ON_DEMAND"
    ) -> str:
        """
        Create Bedrock model import job
        
        Args:
            job_name: Unique job name
            model_name: Name for the imported model
            s3_uri: S3 URI where model files are stored
            role_arn: IAM role ARN with Bedrock import permissions
            inference_type: "ON_DEMAND" or "PROVISIONED"
            
        Returns:
            Job ID
        """
        try:
            logger.info(f"🚀 Creating Bedrock import job: {job_name}")
            logger.info(f"   Model name: {model_name}")
            logger.info(f"   S3 URI: {s3_uri}")
            logger.info(f"   Role ARN: {role_arn}")
            
            response = self.bedrock_client.create_model_import_job(
                jobName=job_name,
                importedModelName=model_name,
                roleArn=role_arn,
                modelDataSource={
                    "s3DataSource": {
                        "s3Uri": s3_uri
                    }
                }
            )
            
            # Debug: Log the full response to see what we actually get
            logger.info(f"📋 Full API response: {response}")
            
            # AWS returns jobArn, extract job ID from it
            job_arn = response.get("jobArn")
            if job_arn:
                # Extract job ID from ARN: arn:aws:bedrock:region:account:model-import-job/JOB_ID
                job_id = job_arn.split('/')[-1]
                logger.info(f"✅ Extracted job ID from ARN: {job_id}")
                return job_id
            
            # Fallback: try other possible field names
            job_id = response.get("jobId") or response.get("JobId") or response.get("jobIdentifier")
            
            if not job_id:
                logger.error(f"❌ No job ID found in response. Available keys: {list(response.keys())}")
                raise KeyError(f"Job ID not found in response. Available keys: {list(response.keys())}")
            
            logger.info(f"✅ Import job created with ID: {job_id}")
            return job_id
            
        except ClientError as e:
            logger.error(f"Failed to create import job: {e}")
            raise
    
    def wait_for_import_completion(self, job_id: str, timeout_minutes: int = 60) -> Dict[str, Any]:
        """
        Wait for import job to complete
        
        Args:
            job_id: Import job ID
            timeout_minutes: Maximum time to wait
            
        Returns:
            Job status information
        """
        logger.info(f"⏳ Waiting for import job {job_id} to complete...")
        
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        while True:
            try:
                response = self.bedrock_client.get_model_import_job(jobIdentifier=job_id)
                status = response["status"]
                
                logger.info(f"   Job status: {status}")
                
                if status == "COMPLETED":
                    model_id = response.get("modelId")
                    logger.info(f"✅ Import completed successfully!")
                    logger.info(f"🎯 Model ID: {model_id}")
                    return response
                
                elif status == "FAILED":
                    failure_message = response.get("failureMessage", "Unknown error")
                    logger.error(f"❌ Import failed: {failure_message}")
                    raise RuntimeError(f"Import job failed: {failure_message}")
                
                elif status in ["IN_PROGRESS", "PENDING"]:
                    # Check timeout
                    elapsed = time.time() - start_time
                    if elapsed > timeout_seconds:
                        raise TimeoutError(f"Import job timed out after {timeout_minutes} minutes")
                    
                    # Wait before checking again
                    time.sleep(30)
                
                else:
                    logger.warning(f"Unknown status: {status}")
                    time.sleep(30)
                    
            except ClientError as e:
                logger.error(f"Error checking job status: {e}")
                raise
    
    def test_model_inference(self, model_id: str, test_prompt: str = "Hello, how are you?") -> str:
        """
        Test the imported model with a simple inference call
        
        Args:
            model_id: Bedrock model ID
            test_prompt: Test prompt to send
            
        Returns:
            Model response
        """
        try:
            logger.info(f"🧪 Testing model inference: {model_id}")
            logger.info(f"   Test prompt: {test_prompt}")
            
            # Prepare request body (format may vary by model)
            request_body = {
                "inputText": test_prompt,
                "textGenerationConfig": {
                    "maxTokenCount": 100,
                    "temperature": 0.7
                }
            }
            
            response = self.bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body),
                contentType="application/json"
            )
            
            response_body = json.loads(response["body"].read().decode())
            generated_text = response_body.get("results", [{}])[0].get("outputText", "")
            
            logger.info(f"✅ Model response: {generated_text}")
            return generated_text
            
        except ClientError as e:
            logger.error(f"Failed to test model: {e}")
            raise
    
    def import_model_to_bedrock(
        self,
        model_path: str,
        model_name: str,
        s3_bucket: str,
        role_arn: str,
        s3_prefix: Optional[str] = None,
        job_name: Optional[str] = None,
        test_inference: bool = True
    ) -> Dict[str, Any]:
        """
        Complete workflow to import a model to Bedrock
        
        Args:
            model_path: Local path to trained model
            model_name: Name for the Bedrock model
            s3_bucket: S3 bucket for model storage
            role_arn: IAM role ARN with Bedrock permissions
            s3_prefix: S3 prefix (defaults to model_name)
            job_name: Import job name (defaults to model_name + timestamp)
            test_inference: Whether to test the model after import
            
        Returns:
            Dictionary with import results
        """
        # Set defaults
        if s3_prefix is None:
            s3_prefix = f"bedrock-models/{model_name}"
        
        if job_name is None:
            timestamp = int(time.time())
            # Replace underscores with hyphens to comply with AWS regex pattern
            safe_model_name = model_name.replace('_', '-')
            job_name = f"{safe_model_name}-import-{timestamp}"
        
        results = {
            "model_name": model_name,
            "job_name": job_name,
            "s3_uri": None,
            "job_id": None,
            "model_id": None,
            "status": "STARTED"
        }
        
        try:
            # Step 1: Validate model
            logger.info("🔍 Step 1: Validating model...")
            if not self.validate_model_architecture(model_path):
                raise ValueError("Model architecture validation failed")
            
            validation = self.validate_model_files(model_path)
            logger.info(f"   File validation: {validation}")
            
            # Step 2: Upload to S3
            logger.info("📤 Step 2: Uploading to S3...")
            s3_uri = self.upload_model_to_s3(model_path, s3_bucket, s3_prefix)
            results["s3_uri"] = s3_uri
            
            # Step 3: Create import job
            logger.info("🚀 Step 3: Creating import job...")
            job_id = self.create_import_job(job_name, model_name, s3_uri, role_arn)
            results["job_id"] = job_id
            
            # Step 4: Wait for completion
            logger.info("⏳ Step 4: Waiting for completion...")
            job_response = self.wait_for_import_completion(job_id)
            results["model_id"] = job_response.get("modelId")
            results["status"] = "COMPLETED"
            
            # Step 5: Test inference (optional)
            if test_inference and results["model_id"]:
                logger.info("🧪 Step 5: Testing inference...")
                try:
                    test_result = self.test_model_inference(results["model_id"])
                    results["test_response"] = test_result
                except Exception as e:
                    logger.warning(f"Inference test failed (this is optional): {e}")
                    results["test_response"] = f"Test failed: {e}"
            
            logger.info("🎉 Bedrock import completed successfully!")
            return results
            
        except Exception as e:
            results["status"] = "FAILED"
            results["error"] = str(e)
            logger.error(f"❌ Bedrock import failed: {e}")
            raise


def create_bedrock_iam_role_guide() -> str:
    """
    Return instructions for creating the required IAM role
    """
    return """
    🔧 AWS Bedrock IAM Role Setup Guide
    ====================================
    
    You need an IAM role with the following permissions:
    
    1. Create IAM Role:
       - Service: bedrock.amazonaws.com
       - Use case: Bedrock Model Import
    
    2. Attach Policy (inline policy):
    
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:ListBucket"
                ],
                "Resource": [
                    "arn:aws:s3:::your-bucket/*",
                    "arn:aws:s3:::your-bucket"
                ]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock:CreateModelImportJob",
                    "bedrock:GetModelImportJob"
                ],
                "Resource": "*"
            }
        ]
    }
    
    3. Trust Relationship:
    
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "bedrock.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    4. Note the Role ARN: arn:aws:iam::YOUR-ACCOUNT:role/BedrockImportRole
    """ 