"""
Agentic RAG Implementation using LangGraph
Based on: https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import click

# LangGraph imports
try:
    from langgraph.graph import StateGraph, START, END, MessagesState
    from langgraph.prebuilt import ToolNode, tools_condition
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    from langchain.tools.retriever import create_retriever_tool
    from langchain.chat_models import init_chat_model
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    MessagesState = None

# IdeaWeaver imports
from .rag import RAGManager


@dataclass
class AgenticRAGConfig:
    """Configuration for Agentic RAG"""
    kb_name: str
    llm_model: str = "openai:gpt-4"
    temperature: float = 0.0
    grading_threshold: float = 0.7
    max_retries: int = 2
    verbose: bool = False


class AgenticRAGManager:
    """
    Agentic RAG Manager with intelligent retrieval decisions
    Implements the LangGraph-based workflow for enhanced RAG
    """
    
    def __init__(self, rag_manager: RAGManager, config: AgenticRAGConfig):
        """
        Initialize Agentic RAG Manager
        
        Args:
            rag_manager: Traditional RAG manager instance
            config: Agentic RAG configuration
        """
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph not available. Install with: pip install langgraph"
            )
        
        self.rag_manager = rag_manager
        self.config = config
        self.verbose = config.verbose
        
        # Initialize chat model
        try:
            self.chat_model = init_chat_model(
                config.llm_model, 
                temperature=config.temperature
            )
        except Exception as e:
            if self.verbose:
                click.echo(f"⚠️  Failed to initialize {config.llm_model}, falling back to gpt-3.5-turbo")
            self.chat_model = init_chat_model("openai:gpt-3.5-turbo", temperature=config.temperature)
        
        # Initialize retriever tool
        self.retriever_tool = None
        self.graph = None
        
        if self.verbose:
            click.echo(f"🤖 Agentic RAG Manager initialized")
            click.echo(f"   LLM: {config.llm_model}")
            click.echo(f"   Knowledge Base: {config.kb_name}")
    
    def setup_retriever_tool(self):
        """Setup the retriever tool from the knowledge base"""
        try:
            # Get the vector store from RAG manager
            kb_config = self.rag_manager.get_knowledge_base(self.config.kb_name)
            if not kb_config:
                raise RuntimeError(f"Knowledge base '{self.config.kb_name}' not found")
            
            vector_store = self.rag_manager._init_vector_store(
                kb_config.name, 
                kb_config.embedding_model
            )
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            
            # Create retriever tool
            self.retriever_tool = create_retriever_tool(
                retriever,
                f"retrieve_{self.config.kb_name}",
                f"Search and return information from {self.config.kb_name} knowledge base."
            )
            
            if self.verbose:
                click.echo(f"🔧 Retriever tool created for KB: {self.config.kb_name}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to setup retriever tool: {e}")
    
    def generate_query_or_respond(self, state: MessagesState) -> Dict[str, List]:
        """
        Agent decides whether to retrieve documents or respond directly
        """
        if self.verbose:
            click.echo("🧠 Agent deciding: retrieve or respond directly...")
        
        response = (
            self.chat_model
            .bind_tools([self.retriever_tool])
            .invoke(state["messages"])
        )
        
        if response.tool_calls:
            if self.verbose:
                click.echo(f"🔍 Agent decided to retrieve: {response.tool_calls[0]['args']['query']}")
        else:
            if self.verbose:
                click.echo("💬 Agent decided to respond directly")
        
        return {"messages": [response]}
    
    def grade_documents(self, state: MessagesState) -> str:
        """
        Grade retrieved documents for relevance to the question
        """
        if self.verbose:
            click.echo("📊 Grading document relevance...")
        
        # Get the original question and retrieved documents
        question = state["messages"][0].content
        last_message = state["messages"][-1]
        
        if not hasattr(last_message, 'content') or not last_message.content:
            if self.verbose:
                click.echo("❌ No documents to grade")
            return "rewrite_question"
        
        documents = last_message.content
        
        # Grade prompt
        GRADE_PROMPT = f"""
        You are a grader assessing relevance of retrieved documents to a user question.
        
        Here is the retrieved document:
        {documents}
        
        Here is the user question:
        {question}
        
        If the document contains information related to the user question, grade it as relevant.
        Give a binary score 'yes' or 'no' to indicate whether the document is relevant.
        
        Respond with only 'yes' or 'no'.
        """
        
        grade_response = self.chat_model.invoke([{"role": "user", "content": GRADE_PROMPT}])
        grade = grade_response.content.strip().lower()
        
        if self.verbose:
            click.echo(f"📊 Document relevance grade: {grade}")
        
        if grade == "yes":
            return "generate_answer"
        else:
            return "rewrite_question"
    
    def rewrite_question(self, state: MessagesState) -> Dict[str, List]:
        """
        Rewrite the original question for better retrieval
        """
        if self.verbose:
            click.echo("✍️  Rewriting question for better retrieval...")
        
        question = state["messages"][0].content
        
        REWRITE_PROMPT = f"""
        You are a question re-writer that converts an input question to a better version that is optimized
        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.
        
        Here is the initial question:
        {question}
        
        Formulate an improved question that is more specific and likely to retrieve relevant documents:
        """
        
        response = self.chat_model.invoke([{"role": "user", "content": REWRITE_PROMPT}])
        
        if self.verbose:
            click.echo(f"✍️  Rewritten question: {response.content}")
        
        return {"messages": [HumanMessage(content=response.content)]}
    
    def generate_answer(self, state: MessagesState) -> Dict[str, List]:
        """
        Generate final answer based on question and retrieved context
        """
        if self.verbose:
            click.echo("📝 Generating final answer...")
        
        question = state["messages"][0].content
        context = state["messages"][-1].content
        
        GENERATE_PROMPT = f"""
        You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        
        Question: {question}
        Context: {context}
        
        Answer:
        """
        
        response = self.chat_model.invoke([{"role": "user", "content": GENERATE_PROMPT}])
        
        if self.verbose:
            click.echo("✅ Answer generated successfully")
        
        return {"messages": [response]}
    
    def build_graph(self):
        """
        Build the Agentic RAG workflow graph
        """
        if self.verbose:
            click.echo("🏗️  Building Agentic RAG workflow graph...")
        
        workflow = StateGraph(MessagesState)
        
        # Add nodes
        workflow.add_node("generate_query_or_respond", self.generate_query_or_respond)
        workflow.add_node("retrieve", ToolNode([self.retriever_tool]))
        workflow.add_node("rewrite_question", self.rewrite_question)
        workflow.add_node("generate_answer", self.generate_answer)
        
        # Add edges
        workflow.add_edge(START, "generate_query_or_respond")
        
        # Conditional edge: decide whether to retrieve or respond
        workflow.add_conditional_edges(
            "generate_query_or_respond",
            tools_condition,
            {
                "tools": "retrieve",
                END: END,
            },
        )
        
        # Conditional edge: grade documents and decide next step
        workflow.add_conditional_edges(
            "retrieve",
            self.grade_documents,
            {
                "generate_answer": "generate_answer",
                "rewrite_question": "rewrite_question",
            }
        )
        
        # Connect final edges
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("rewrite_question", "generate_query_or_respond")
        
        # Compile the graph
        self.graph = workflow.compile()
        
        if self.verbose:
            click.echo("✅ Agentic RAG graph compiled successfully")
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the Agentic RAG system
        
        Args:
            question: User question
            
        Returns:
            Response dictionary with answer and metadata
        """
        if not self.graph:
            self.setup_retriever_tool()
            self.build_graph()
        
        if self.verbose:
            click.echo(f"🤖 Agentic RAG processing: {question}")
        
        # Run the graph
        result = self.graph.invoke({
            "messages": [HumanMessage(content=question)]
        })
        
        # Extract the final answer
        final_message = result["messages"][-1]
        answer = final_message.content if hasattr(final_message, 'content') else str(final_message)
        
        # Trace the workflow
        workflow_trace = []
        for message in result["messages"]:
            if hasattr(message, 'tool_calls') and message.tool_calls:
                workflow_trace.append("🔍 Retrieved documents")
            elif isinstance(message, ToolMessage):
                workflow_trace.append("📄 Document grading")
            elif "improved question" in str(message).lower():
                workflow_trace.append("✍️ Question rewritten")
        
        return {
            "answer": answer,
            "question": question,
            "workflow_trace": workflow_trace,
            "message_count": len(result["messages"]),
            "rag_type": "agentic"
        }
    
    def visualize_graph(self) -> str:
        """
        Get a text representation of the workflow graph
        """
        if not self.graph:
            return "Graph not built yet. Call query() first."
        
        return """
🤖 Agentic RAG Workflow:

[START] → [Agent Decision] 
                ↓
        [Retrieve?] → [Respond Directly] → [END]
                ↓
        [Get Documents] → [Grade Relevance]
                ↓                    ↓
        [Relevant?] → [Generate Answer] → [END]
                ↓
        [Rewrite Question] → [Agent Decision]
        """ 