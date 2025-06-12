"""
RAG Evaluation Module using RAGAS Framework
"""

import os
import json
import time
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import click

# Always import Dataset since it's from datasets library, not RAGAS
try:
    from datasets import Dataset
except ImportError:
    Dataset = None

# RAGAS imports
try:
    from ragas import evaluate
    from ragas.metrics import (
        context_precision,
        context_recall, 
        faithfulness,
        answer_relevancy,
        answer_correctness,
        answer_similarity
    )
    # Try to import LLM configuration
    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_community.llms import Ollama
        from langchain_community.embeddings import OllamaEmbeddings
        LOCAL_LLM_AVAILABLE = True
    except ImportError:
        LOCAL_LLM_AVAILABLE = False
    
    RAGAS_AVAILABLE = True
except ImportError as e:
    RAGAS_AVAILABLE = False
    LOCAL_LLM_AVAILABLE = False

# IdeaWeaver imports
from .rag import RAGManager


@dataclass
class RAGEvaluationConfig:
    """Configuration for RAG evaluation"""
    kb_name: str
    test_questions: List[str]
    ground_truths: Optional[List[List[str]]] = None
    llm_model: Optional[str] = None
    local_llm: Optional[str] = None
    openai_key: Optional[str] = None
    output_dir: str = "./rag_evaluations"
    metrics: List[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            # Use metrics that don't require ground truth by default
            self.metrics = ["faithfulness", "answer_relevancy"]
            if self.ground_truths:
                self.metrics.append("context_recall")


@dataclass 
class RAGEvaluationResult:
    """Results from RAG evaluation"""
    kb_name: str
    evaluation_id: str
    timestamp: str
    metrics_scores: Dict[str, float]
    detailed_results: pd.DataFrame
    summary_report: str
    config: RAGEvaluationConfig
    
    def to_dict(self) -> Dict:
        return {
            "kb_name": self.kb_name,
            "evaluation_id": self.evaluation_id,
            "timestamp": self.timestamp,
            "metrics_scores": self.metrics_scores,
            "summary_report": self.summary_report,
            "config": asdict(self.config)
        }


class RAGEvaluator:
    """
    RAG Evaluation using RAGAS framework
    Based on: https://medium.com/data-science/evaluating-rag-applications-with-ragas-81d67b0ee31a
    """
    
    def __init__(self, rag_manager: RAGManager, verbose: bool = False):
        """
        Initialize RAG Evaluator
        
        Args:
            rag_manager: RAGManager instance
            verbose: Enable verbose logging
        """
        if not RAGAS_AVAILABLE:
            raise ImportError(
                "RAGAS not available. Install with: pip install ragas datasets"
            )
        
        self.rag_manager = rag_manager
        self.verbose = verbose
        
        # Create evaluations directory
        self.eval_dir = Path("./rag_evaluations")
        self.eval_dir.mkdir(exist_ok=True)
        
        if self.verbose:
            click.echo(f"📊 RAG Evaluator initialized")
            click.echo(f"📁 Evaluations directory: {self.eval_dir}")
    
    def prepare_evaluation_dataset(self, config: RAGEvaluationConfig) -> Any:
        """
        Prepare evaluation dataset by running RAG pipeline on test questions
        
        Args:
            config: Evaluation configuration
            
        Returns:
            RAGAS-compatible dataset
        """
        if self.verbose:
            click.echo(f"🔍 Preparing evaluation dataset for KB: {config.kb_name}")
            click.echo(f"❓ Test questions: {len(config.test_questions)}")
        
        questions = config.test_questions
        answers = []
        contexts = []
        
        # Run RAG pipeline for each question
        for i, question in enumerate(questions):
            if self.verbose:
                click.echo(f"   Processing question {i+1}/{len(questions)}: {question[:50]}...")
            
            try:
                # Query the RAG system
                result = self.rag_manager.query(
                    kb_name=config.kb_name,
                    question=question,
                    top_k=5,  # Get more contexts for evaluation
                    llm_model=config.llm_model
                )
                
                # Extract answer and contexts
                if config.llm_model and result.get("answer"):
                    answers.append(result["answer"])
                else:
                    # If no LLM, create a simple answer from retrieved contexts
                    retrieved_contexts = [doc["content"] for doc in result["retrieved_documents"]]
                    answers.append(f"Based on the context: {retrieved_contexts[0][:200]}...")
                
                # Extract contexts
                contexts.append([doc["content"] for doc in result["retrieved_documents"]])
                
            except Exception as e:
                if self.verbose:
                    click.echo(f"   ⚠️  Error processing question: {e}")
                answers.append("Error generating answer")
                contexts.append(["No context retrieved"])
        
        # Prepare dataset dictionary
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts
        }
        
        # Add ground truths if provided
        if config.ground_truths:
            if len(config.ground_truths) != len(questions):
                raise ValueError("Number of ground truths must match number of questions")
            data["ground_truths"] = config.ground_truths
        
        # Convert to RAGAS dataset
        dataset = Dataset.from_dict(data)
        
        if self.verbose:
            click.echo(f"✅ Dataset prepared with {len(dataset)} samples")
        
        return dataset
    
    def get_ragas_metrics(self, metric_names: List[str]) -> List:
        """Get RAGAS metric objects from names"""
        metric_map = {
            "context_precision": context_precision,
            "context_recall": context_recall,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "answer_correctness": answer_correctness,
            "answer_similarity": answer_similarity
        }
        
        metrics = []
        for name in metric_names:
            if name in metric_map:
                metrics.append(metric_map[name])
            else:
                if self.verbose:
                    click.echo(f"⚠️  Unknown metric: {name}")
        
        return metrics
    
    def evaluate_rag_pipeline(self, config: RAGEvaluationConfig) -> RAGEvaluationResult:
        """
        Evaluate RAG pipeline using RAGAS metrics
        
        Args:
            config: Evaluation configuration
            
        Returns:
            Evaluation results
        """
        if self.verbose:
            click.echo(f"🧪 Starting RAG evaluation with RAGAS")
            click.echo(f"📋 Metrics: {', '.join(config.metrics)}")
        
        # Prepare evaluation dataset
        dataset = self.prepare_evaluation_dataset(config)
        
        # Get RAGAS metrics
        metrics = self.get_ragas_metrics(config.metrics)
        
        if not metrics:
            raise ValueError("No valid metrics specified")
        
        # Run RAGAS evaluation
        if self.verbose:
            click.echo(f"⚡ Running RAGAS evaluation...")
        
        start_time = time.time()
        
        try:
            # Try to configure local LLMs first
            local_llm, local_embeddings = self.configure_local_llms()
            
            # Check if OpenAI API key is available
            openai_key = os.environ.get('OPENAI_API_KEY')
            
            if not openai_key and not local_llm:
                # Fallback: use only metrics that don't require LLM
                if self.verbose:
                    click.echo("⚠️  No OpenAI API key or local LLM available")
                    click.echo("💡 Consider setting OPENAI_API_KEY or installing Ollama")
                    click.echo("🔄 Falling back to basic similarity-based evaluation")
                
                # For now, let's create a simple custom evaluation
                raise RuntimeError("LLM-based evaluation requires either OpenAI API key or local LLM setup")
            
            # Run evaluation with configured LLMs if available
            eval_kwargs = {"dataset": dataset, "metrics": metrics}
            if local_llm and local_embeddings:
                eval_kwargs["llm"] = local_llm
                eval_kwargs["embeddings"] = local_embeddings
            
            result = evaluate(**eval_kwargs)
            
            evaluation_time = time.time() - start_time
            
            if self.verbose:
                click.echo(f"✅ Evaluation completed in {evaluation_time:.2f} seconds")
            
        except Exception as e:
            raise RuntimeError(f"RAGAS evaluation failed: {e}")
        
        # Process results
        df_results = result.to_pandas()
        
        # Calculate average scores
        metrics_scores = {}
        for metric_name in config.metrics:
            if metric_name in df_results.columns:
                score = df_results[metric_name].mean()
                metrics_scores[metric_name] = float(score)
        
        # Generate evaluation ID and timestamp
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        evaluation_id = f"eval_{config.kb_name}_{int(time.time())}"
        
        # Create summary report
        summary_report = self._generate_summary_report(
            config, metrics_scores, df_results, evaluation_time
        )
        
        # Create result object
        eval_result = RAGEvaluationResult(
            kb_name=config.kb_name,
            evaluation_id=evaluation_id,
            timestamp=timestamp,
            metrics_scores=metrics_scores,
            detailed_results=df_results,
            summary_report=summary_report,
            config=config
        )
        
        # Save results
        self._save_evaluation_results(eval_result)
        
        return eval_result
    
    def _generate_summary_report(self, config: RAGEvaluationConfig, 
                                metrics_scores: Dict[str, float],
                                detailed_results: pd.DataFrame,
                                evaluation_time: float) -> str:
        """Generate human-readable summary report"""
        
        report = f"""# RAG Evaluation Report

## 📊 Overview
- **Knowledge Base**: {config.kb_name}
- **Test Questions**: {len(config.test_questions)}
- **Evaluation Time**: {evaluation_time:.2f} seconds
- **Timestamp**: {config.test_questions}

## 🎯 RAGAS Metrics Scores

"""
        
        # Add metric scores with interpretations
        metric_interpretations = {
            "context_precision": "Signal-to-noise ratio of retrieved context",
            "context_recall": "Coverage of relevant information in retrieved context",
            "faithfulness": "Factual accuracy of generated answers",
            "answer_relevancy": "Relevance of answers to questions",
            "answer_correctness": "Overall correctness of answers",
            "answer_similarity": "Semantic similarity to ground truth"
        }
        
        for metric, score in metrics_scores.items():
            interpretation = metric_interpretations.get(metric, "")
            performance = self._get_performance_label(score)
            
            report += f"### {metric.replace('_', ' ').title()}\n"
            report += f"- **Score**: {score:.3f} {performance}\n"
            report += f"- **Description**: {interpretation}\n\n"
        
        # Add overall assessment
        avg_score = sum(metrics_scores.values()) / len(metrics_scores)
        overall_performance = self._get_performance_label(avg_score)
        
        report += f"""## 📈 Overall Assessment
- **Average Score**: {avg_score:.3f} {overall_performance}
- **Best Metric**: {max(metrics_scores.items(), key=lambda x: x[1])[0]} ({max(metrics_scores.values()):.3f})
- **Needs Improvement**: {min(metrics_scores.items(), key=lambda x: x[1])[0]} ({min(metrics_scores.values()):.3f})

## 💡 Recommendations

"""
        
        # Add recommendations based on scores
        recommendations = []
        
        if metrics_scores.get("context_precision", 1.0) < 0.7:
            recommendations.append("🔍 **Improve Retrieval Precision**: Consider tuning chunk size, using hybrid search, or improving query preprocessing")
        
        if metrics_scores.get("context_recall", 1.0) < 0.7:
            recommendations.append("📚 **Improve Context Coverage**: Increase retrieved documents count or improve chunking strategy")
        
        if metrics_scores.get("faithfulness", 1.0) < 0.8:
            recommendations.append("✅ **Improve Answer Faithfulness**: Fine-tune LLM prompts or use better instruction-following models")
        
        if metrics_scores.get("answer_relevancy", 1.0) < 0.8:
            recommendations.append("🎯 **Improve Answer Relevancy**: Optimize prompts to ensure answers directly address questions")
        
        if not recommendations:
            recommendations.append("🎉 **Excellent Performance**: Your RAG pipeline is performing well across all metrics!")
        
        for rec in recommendations:
            report += f"- {rec}\n"
        
        report += f"""

## 🔧 Next Steps
1. **Review detailed results** in the exported CSV file
2. **Implement recommended improvements**
3. **Re-evaluate** to measure progress
4. **A/B test** different configurations

---
*Generated by IdeaWeaver RAG Evaluator using RAGAS framework*
"""
        
        return report
    
    def _get_performance_label(self, score: float) -> str:
        """Get performance label emoji based on score"""
        if score >= 0.9:
            return "🟢 Excellent"
        elif score >= 0.8:
            return "🟡 Good"
        elif score >= 0.7:
            return "🟠 Fair"
        else:
            return "🔴 Needs Improvement"
    
    def _save_evaluation_results(self, result: RAGEvaluationResult):
        """Save evaluation results to files"""
        eval_path = self.eval_dir / result.evaluation_id
        eval_path.mkdir(exist_ok=True)
        
        # Save summary report
        report_path = eval_path / "summary_report.md"
        with open(report_path, 'w') as f:
            f.write(result.summary_report)
        
        # Save detailed results CSV
        csv_path = eval_path / "detailed_results.csv"
        result.detailed_results.to_csv(csv_path, index=False)
        
        # Save evaluation metadata
        metadata_path = eval_path / "evaluation_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        if self.verbose:
            click.echo(f"💾 Results saved to: {eval_path}")
            click.echo(f"   📄 Summary: {report_path}")
            click.echo(f"   📊 Details: {csv_path}")
    
    def load_test_questions_from_file(self, file_path: str) -> Tuple[List[str], Optional[List[List[str]]]]:
        """Load test questions and ground truths from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        questions = data.get("questions", [])
        ground_truths = data.get("ground_truths")
        
        return questions, ground_truths
    
    def generate_test_questions(self, kb_name: str, num_questions: int = 10) -> List[str]:
        """
        Generate test questions automatically based on knowledge base content
        
        Args:
            kb_name: Knowledge base name
            num_questions: Number of questions to generate
            
        Returns:
            List of generated questions
        """
        # This is a simplified version - in production you'd use LLM for question generation
        
        # Get some sample content from the knowledge base
        sample_result = self.rag_manager.query(
            kb_name=kb_name,
            question="summary",  # Generic query to get content
            top_k=10
        )
        
        # Extract key concepts and generate questions
        questions = [
            "What is the main topic discussed in the documents?",
            "What are the key features mentioned?",
            "How does this work?",
            "What are the benefits?",
            "What are the requirements?",
            "How do you get started?",
            "What are the best practices?",
            "What should you avoid?",
            "Who is the target audience?",
            "What are the next steps?"
        ]
        
        return questions[:num_questions]

    def configure_local_llms(self):
        """Configure RAGAS to use local LLMs instead of OpenAI"""
        if not LOCAL_LLM_AVAILABLE:
            if self.verbose:
                click.echo("⚠️  Local LLM support not available, using default configuration")
            return None, None
        
        try:
            # Try to use Ollama (local LLM)
            local_llm = Ollama(model="llama2")  # You can change this to any Ollama model
            local_embeddings = OllamaEmbeddings(model="llama2")
            
            # Wrap for RAGAS
            wrapped_llm = LangchainLLMWrapper(local_llm)
            wrapped_embeddings = LangchainEmbeddingsWrapper(local_embeddings)
            
            if self.verbose:
                click.echo("🚀 Configured RAGAS to use local Ollama LLM")
            
            return wrapped_llm, wrapped_embeddings
            
        except Exception as e:
            if self.verbose:
                click.echo(f"⚠️  Failed to configure local LLM: {e}")
            return None, None

    def get_simple_metrics(self):
        """Get metrics that don't require LLM evaluation"""
        # Use metrics that can work without complex LLM evaluation
        simple_metrics = []
        
        # These metrics work with basic similarity calculations
        if self.verbose:
            click.echo("📊 Using simplified evaluation metrics (no LLM required)")
        
        return simple_metrics 