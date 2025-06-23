"""
CrewAI Integration Module for IdeaWeaver

This module provides integration with CrewAI for creating and managing AI agent workflows.
It includes base functionality that can be extended for different use cases like storybook generation,
data analysis, content creation, etc.
"""

from typing import List, Dict, Any, Optional
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
import os
from pathlib import Path
import yaml
import logging
import re
import requests
import subprocess

logger = logging.getLogger(__name__)

def setup_intelligent_llm(openai_api_key: Optional[str] = None, preferred_model: str = None) -> tuple:
    """
    Intelligently set up LLM with Ollama as first preference, OpenAI as fallback.
    Returns (llm_instance, llm_type, model_used)
    """
    
    # First try Ollama using CrewAI's LLM wrapper
    try:
        print("🔍 Checking for Ollama availability...")
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            models_data = response.json()
            available_models = [model['name'] for model in models_data.get('models', [])]
            
            if available_models:
                # Use preferred model if specified and available, otherwise use the first available
                model_to_use = None
                
                if preferred_model and preferred_model in available_models:
                    model_to_use = preferred_model
                else:
                    # Smart model selection - prefer smaller/faster models for CrewAI
                    priority_models = ['phi3', 'llama3', 'mistral', 'deepseek']
                    for priority in priority_models:
                        for model in available_models:
                            if priority in model.lower():
                                model_to_use = model
                                break
                        if model_to_use:
                            break
                    
                    # If no priority model found, use the first available
                    if not model_to_use:
                        model_to_use = available_models[0]
                
                print(f"✅ Ollama is available! Using model: {model_to_use}")
                print(f"📋 Available models: {', '.join(available_models)}")
                
                try:
                    from crewai import LLM
                    # Use CrewAI's LLM wrapper - the correct approach!
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
        # No hardcoded key - require user to provide one
        raise RuntimeError("OpenAI API key required. Please set OPENAI_API_KEY environment variable or pass --openai-api-key parameter.")
    
    try:
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
        print("✅ OpenAI setup successful")
        return llm, 'openai', 'gpt-4o-mini'
    except Exception as e:
        print(f"❌ OpenAI setup failed: {e}")
        raise RuntimeError("Both Ollama and OpenAI failed")

def get_available_ollama_models() -> List[str]:
    """Get list of available Ollama models"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            return [model['name'] for model in models_data.get('models', [])]
    except:
        pass
    return []

class CrewAIManager:
    """Base class for managing CrewAI workflows in IdeaWeaver"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the CrewAI manager
        
        Args:
            config_path: Optional path to a YAML configuration file
        """
        self.llm = None
        self.agents = {}
        self.tasks = []
        self.crew = None
        self.config = {}
        
        if config_path:
            self.load_config(config_path)
            
    def load_config(self, config_path: str) -> None:
        """Load configuration from a YAML file"""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            raise
            
    def setup_llm(self, model_name: str = "gpt-4-turbo-preview", **kwargs) -> None:
        """Set up the language model for agents"""
        try:
            self.llm = ChatOpenAI(
                model_name=model_name,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error setting up LLM: {e}")
            raise
            
    def create_agent(self, 
                    name: str,
                    role: str,
                    goal: str,
                    backstory: str,
                    tools: List[Any] = None,
                    verbose: bool = False) -> Agent:
        """Create a new agent with specified parameters"""
        try:
            agent = Agent(
                name=name,
                role=role,
                goal=goal,
                backstory=backstory,
                tools=tools or [],
                verbose=verbose,
                llm=self.llm
            )
            self.agents[name] = agent
            return agent
        except Exception as e:
            logger.error(f"Error creating agent {name}: {e}")
            raise
            
    def create_task(self,
                   description: str,
                   agent: Agent,
                   expected_output: str,
                   context: List[Task] = None) -> Task:
        """Create a new task for an agent"""
        try:
            task = Task(
                description=description,
                agent=agent,
                expected_output=expected_output,
                context=context or []
            )
            self.tasks.append(task)
            return task
        except Exception as e:
            logger.error(f"Error creating task: {e}")
            raise
            
    def create_crew(self,
                   agents: List[Agent],
                   tasks: List[Task],
                   process: Process = Process.sequential,
                   verbose: bool = False) -> Crew:
        """Create a new crew with specified agents and tasks"""
        try:
            self.crew = Crew(
                agents=agents,
                tasks=tasks,
                process=process,
                verbose=verbose
            )
            return self.crew
        except Exception as e:
            logger.error(f"Error creating crew: {e}")
            raise
            
    def run_workflow(self) -> Any:
        """Run the current workflow"""
        if not self.crew:
            raise ValueError("No crew has been created. Create a crew first.")
        try:
            return self.crew.kickoff()
        except Exception as e:
            logger.error(f"Error running workflow: {e}")
            raise
            
    def save_workflow(self, output_path: str) -> None:
        """Save the current workflow configuration"""
        try:
            workflow_config = {
                'agents': {name: {
                    'role': agent.role,
                    'goal': agent.goal,
                    'backstory': agent.backstory
                } for name, agent in self.agents.items()},
                'tasks': [{
                    'description': task.description,
                    'agent': task.agent.name,
                    'expected_output': task.expected_output
                } for task in self.tasks]
            }
            
            with open(output_path, 'w') as f:
                yaml.dump(workflow_config, f)
        except Exception as e:
            logger.error(f"Error saving workflow: {e}")
            raise
            
    def load_workflow(self, workflow_path: str) -> None:
        """Load a workflow configuration"""
        try:
            with open(workflow_path, 'r') as f:
                workflow_config = yaml.safe_load(f)
                
            # Recreate agents
            for name, agent_config in workflow_config['agents'].items():
                self.create_agent(
                    name=name,
                    role=agent_config['role'],
                    goal=agent_config['goal'],
                    backstory=agent_config['backstory']
                )
                
            # Recreate tasks
            for task_config in workflow_config['tasks']:
                agent = self.agents[task_config['agent']]
                self.create_task(
                    description=task_config['description'],
                    agent=agent,
                    expected_output=task_config['expected_output']
                )
        except Exception as e:
            logger.error(f"Error loading workflow: {e}")
            raise

class StorybookGenerator:
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the StorybookGenerator with intelligent LLM selection."""
        self.llm, self.llm_type, self.model_used = setup_intelligent_llm(openai_api_key)
        print(f"🧠 Using {self.llm_type.upper()} with model: {self.model_used}")

    def _setup_openai_llm(self, openai_api_key: Optional[str] = None):
        """Set up OpenAI LLM (removing broken Ollama integration)."""
        
        print("🔄 Setting up OpenAI (Ollama has compatibility issues with CrewAI v0.121.1)...")
        
        # Set up API key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        elif not os.environ.get("OPENAI_API_KEY"):
            # No hardcoded key - require user to provide one
            raise Exception("OpenAI API key required. Please set OPENAI_API_KEY environment variable or pass --openai-api-key parameter.")
        
        try:
            from langchain_openai import ChatOpenAI
            
            llm = ChatOpenAI(
                model="gpt-4",
                temperature=0.7,
                max_tokens=2000
            )
            
            print("🧪 Testing OpenAI connection...")
            test_response = llm.invoke("Hello! Say 'Hi' back.")
            print(f"✅ OpenAI connection successful: {str(test_response)[:50]}...")
            return llm
                
        except Exception as e:
            print(f"❌ Failed to setup OpenAI: {e}")
            raise Exception("OpenAI setup failed. Please check your API key.")

    def create_storybook(self, 
                        theme: str,
                        target_age: str,
                        num_pages: int = 1,
                        style: str = "whimsical") -> Dict:
        """
        Generate a storybook using CrewAI agents.
        
        Args:
            theme: The main theme of the storybook
            target_age: Target age group (e.g., "3-5", "6-8", "9-12")
            num_pages: Number of pages in the storybook
            style: Writing style (e.g., "whimsical", "educational", "adventure")
            
        Returns:
            Dict containing the generated storybook content
        """
        # Create the story writer agent with better configuration
        writer = Agent(
            role='Expert Children\'s Story Writer',
            goal=f'Create an engaging, {style} story about {theme} perfect for {target_age} year olds',
            backstory=f"""You are a renowned children's book author with over 20 years of experience. 
            You specialize in creating {style} stories that captivate young minds while teaching valuable lessons.
            You understand child psychology and development for ages {target_age}, and you know exactly 
            how to use age-appropriate vocabulary, sentence structure, and themes that resonate with this age group.
            Your stories always have a clear moral or lesson woven naturally into the narrative.""",
            verbose=False,
            llm=self.llm
        )

        # Create the illustrator agent with better configuration
        illustrator = Agent(
            role='Children\'s Book Art Director',
            goal='Create vivid, detailed illustration concepts that bring the story to life',
            backstory=f"""You are a world-class children's book illustrator and art director with an eye for 
            creating magical, engaging visuals. You specialize in {style} illustration styles that appeal 
            to {target_age} year olds. You understand color psychology, composition, and how to create 
            illustrations that support and enhance the story narrative. Your illustration descriptions 
            are so detailed that any artist could recreate them perfectly.""",
            verbose=False,
            llm=self.llm
        )

        # Create the editor agent with better configuration
        editor = Agent(
            role='Senior Children\'s Book Editor',
            goal='Polish and perfect the storybook for publication quality',
            backstory=f"""You are a senior editor at a prestigious children's publishing house with 15+ years 
            of experience. You have an exceptional eye for detail and understand what makes children's books 
            successful. You ensure perfect age-appropriateness for {target_age} year olds, consistent tone, 
            proper pacing, and seamless integration of text and illustration concepts. You transform good 
            stories into unforgettable masterpieces.""",
            verbose=False,
            llm=self.llm
        )

        # Create tasks with better structure and dependencies
        writing_task = Task(
            description=f"""Write a complete {style} children's storybook about {theme} for {target_age} year olds.

            REQUIREMENTS:
            - Exactly {num_pages} pages
            - Age-appropriate vocabulary for {target_age} year olds
            - Clear story structure: engaging opening, exciting middle, satisfying conclusion
            - Memorable characters with distinct personalities
            - A meaningful lesson or positive message
            - Rich sensory details and emotions
            - Perfect pacing for the target age group

            FORMAT:
            Page 1: [Story text for page 1]
            {'Page 2: [Story text for page 2]' if num_pages > 1 else ''}
            {'Page 3: [Story text for page 3]' if num_pages > 2 else ''}
            {'... and so on for all pages' if num_pages > 3 else ''}

            Make the story captivating, with each page ending in a way that makes children want to turn to the next page.
            Include natural dialogue and action that keeps young readers engaged.""",
            agent=writer,
            expected_output=f"A complete {num_pages}-page storybook with engaging text formatted clearly by page number, featuring age-appropriate language and a compelling narrative."
        )

        illustration_task = Task(
            description=f"""Create detailed, professional illustration descriptions for each page of the storybook.

            For each page, provide:
            - Detailed visual description of the scene
            - Character positioning and expressions
            - Setting details and atmosphere
            - Color palette suggestions
            - Mood and emotional tone of the illustration
            - Specific artistic style notes for {style} aesthetic

            FORMAT:
            Page 1 Illustration: [Detailed description]
            {'Page 2 Illustration: [Detailed description]' if num_pages > 1 else ''}
            {'Page 3 Illustration: [Detailed description]' if num_pages > 2 else ''}
            {'... for all pages' if num_pages > 3 else ''}

            Ensure illustrations complement and enhance the story, are appropriate for {target_age} year olds,
            and maintain visual consistency throughout the book.""",
            agent=illustrator,
            expected_output=f"Professional illustration descriptions for all {num_pages} pages with detailed visual specifications.",
            context=[writing_task]  # This task depends on the writing task
        )

        editing_task = Task(
            description=f"""Edit and polish the complete storybook to publication standards.

            EDITING CHECKLIST:
            1. Verify age-appropriateness for {target_age} year olds
            2. Ensure consistent {style} tone throughout
            3. Check story pacing and flow
            4. Verify vocabulary level is appropriate
            5. Ensure illustrations match and enhance the text
            6. Check for any errors or inconsistencies
            7. Optimize readability and engagement
            8. Ensure the lesson/message is clear but not preachy

            FINAL OUTPUT FORMAT:
            Title: [Creative title for the storybook]
            
            Theme: {theme}
            Target Age: {target_age} years old
            Style: {style}
            Pages: {num_pages}
            
            [Complete story with page numbers and corresponding illustration descriptions]
            
            Summary: [Brief description of the story and its message]""",
            agent=editor,
            expected_output="A polished, publication-ready storybook with title, complete text, illustration descriptions, and summary.",
            context=[writing_task, illustration_task]  # This task depends on both previous tasks
        )

        # Create the crew with sequential process for better quality
        crew = Crew(
            agents=[writer, illustrator, editor],
            tasks=[writing_task, illustration_task, editing_task],
            process=Process.sequential,  # Ensure tasks run in order
            verbose=True
        )

        # Execute the crew's tasks
        print(f"\n🎨 Creating your {style} storybook about {theme} for {target_age} year olds...\n")
        
        try:
            result = crew.kickoff()
            
            # Clean up the raw result thoroughly
            clean_result = self._clean_content(str(result))
            
            # Format the final output nicely
            formatted_result = self._format_output(clean_result, theme, target_age, style, num_pages)
            
            print("\n" + "="*80)
            print("🎉 STORYBOOK GENERATION COMPLETE! 🎉")
            print("="*80)
            print(formatted_result)
            print("="*80)
            
            return {
                "theme": theme,
                "target_age": target_age,
                "style": style,
                "num_pages": num_pages,
                "content": clean_result,
                "formatted_content": self._clean_content(formatted_result)
            }
            
        except Exception as e:
            print(f"\n❌ Error during storybook generation: {e}")
            return {
                "theme": theme,
                "target_age": target_age,
                "style": style,
                "num_pages": num_pages,
                "error": str(e)
            }
    
    def _clean_content(self, content: str) -> str:
        """Thoroughly clean content of all escape characters and formatting issues."""
        if not content:
            return ""
            
        clean_content = str(content)
        
        # Replace all common escape characters
        clean_content = clean_content.replace('\\n', '\n')
        clean_content = clean_content.replace('\\r', '\r')
        clean_content = clean_content.replace('\\t', '\t')
        clean_content = clean_content.replace('\\"', '"')
        clean_content = clean_content.replace("\\'", "'")
        clean_content = clean_content.replace('\\\\', '\\')
        
        # Remove any remaining escape sequences
        clean_content = re.sub(r'\\(.)', r'\1', clean_content)
        
        # Normalize whitespace
        clean_content = re.sub(r'\s+', ' ', clean_content)  # Multiple spaces to single space
        clean_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', clean_content)  # Multiple newlines to double
        clean_content = clean_content.strip()
        
        return clean_content
    
    def _format_output(self, content: str, theme: str, target_age: str, style: str, num_pages: int) -> str:
        """Format the output in a nice, readable way."""
        
        # Clean the content first
        clean_content = self._clean_content(content)
        
        # Create a simple, clean header
        header = f"""📚 STORYBOOK GENERATED 📚

Theme: {theme}
Target Age: {target_age} years old
Style: {style}
Pages: {num_pages}

{"-" * 60}

"""
        
        # Add proper formatting with emojis and spacing
        formatted_content = clean_content
        
        # Add emojis and spacing for different sections
        formatted_content = re.sub(r'^Title:', '📚 Title:', formatted_content, flags=re.MULTILINE)
        formatted_content = re.sub(r'^Page (\d+):', r'\n📖 Page \1:', formatted_content, flags=re.MULTILINE)
        formatted_content = re.sub(r'^Illustration:', '\n🎨 Illustration:', formatted_content, flags=re.MULTILINE)
        formatted_content = re.sub(r'^Summary:', '\n📝 Summary:', formatted_content, flags=re.MULTILINE)
        
        # Final cleanup
        result = header + formatted_content
        return self._clean_content(result)

class ResearchWriterGenerator:
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the ResearchWriterGenerator with intelligent LLM selection."""
        self.llm, self.llm_type, self.model_used = setup_intelligent_llm(openai_api_key)
        print(f"🧠 Using {self.llm_type.upper()} with model: {self.model_used}")

    def _setup_openai_llm(self, openai_api_key: Optional[str] = None):
        """Set up OpenAI LLM for research and writing tasks."""
        
        print("🔄 Setting up OpenAI for Research & Writing agents...")
        
        # Set up API key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        elif not os.environ.get("OPENAI_API_KEY"):
            # No hardcoded key - require user to provide one
            raise Exception("OpenAI API key required. Please set OPENAI_API_KEY environment variable or pass --openai-api-key parameter.")
        
        try:
            from langchain_openai import ChatOpenAI
            
            llm = ChatOpenAI(
                model="gpt-4",
                temperature=0.7,
                max_tokens=3000
            )
            
            print("🧪 Testing OpenAI connection...")
            test_response = llm.invoke("Hello! Say 'Hi' back.")
            print(f"✅ OpenAI connection successful: {str(test_response)[:50]}...")
            return llm
                
        except Exception as e:
            print(f"❌ Failed to setup OpenAI: {e}")
            raise Exception("OpenAI setup failed. Please check your API key.")

    def create_research_content(self, 
                              topic: str,
                              content_type: str = "blog post",
                              target_audience: str = "tech enthusiasts") -> Dict:
        """Create research-based content using AI agents."""
        
        print(f"🔍 Creating {content_type} about '{topic}' for {target_audience}")
        print("🧠 Using OpenAI GPT-4 with comprehensive knowledge base")

        # Create the researcher agent (without custom tools, relying on LLM knowledge)
        researcher = Agent(
            role='Senior Research Analyst',
            goal=f'Provide comprehensive, up-to-date analysis and insights about {topic}',
            backstory=f"""You are an expert research analyst with a PhD in your field and over 15 years of experience. 
            You excel at identifying key trends, analyzing complex data, and synthesizing information from multiple sources.
            You have extensive knowledge about current developments, technologies, and industry trends.
            Your research is thorough, methodical, and always includes the most current developments available in your knowledge base.
            You provide well-sourced, accurate information with specific examples, data points, and industry insights.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

        # Create the writer agent
        writer = Agent(
            role='Expert Content Strategist',
            goal=f'Craft compelling, well-structured {content_type} about {topic} for {target_audience}',
            backstory=f"""You are a world-class content strategist and writer with expertise in making complex topics 
            accessible and engaging. You have 20+ years of experience writing for {target_audience} and understand 
            exactly how to structure content for maximum impact and readability. Your writing is clear, engaging, 
            well-organized, and always includes proper formatting with headers, bullet points, and clear sections.""",
            verbose=True,
            allow_delegation=True,
            llm=self.llm
        )

        # Create the editor agent for better formatting
        editor = Agent(
            role='Senior Content Editor',
            goal='Polish and format content to publication standards with excellent readability',
            backstory="""You are a senior editor at a prestigious publication with 15+ years of experience. 
            You excel at taking good content and making it exceptional through proper formatting, clear structure, 
            engaging headlines, and professional presentation. You ensure content is well-organized with proper 
            sections, bullet points, and formatting that enhances readability.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

        # Create research task
        research_task = Task(
            description=f"""Conduct comprehensive research on {topic}. 
            
            RESEARCH REQUIREMENTS:
            - Find the latest developments, trends, and key information
            - Identify major players, technologies, and market impacts
            - Look for recent news, studies, and expert opinions
            - Gather factual data, statistics, and examples
            - Focus on credible sources and current information
            
            DELIVERABLE:
            Provide a detailed research report with:
            - Executive summary of key findings
            - Major trends and developments
            - Key players and technologies
            - Market impact and future outlook
            - Supporting data and examples
            - Source references where applicable""",
            agent=researcher,
            expected_output="A comprehensive research report with key findings, trends, and supporting data organized in clear sections."
        )

        # Create writing task
        writing_task = Task(
            description=f"""Create an engaging {content_type} about {topic} for {target_audience} using the research findings.
            
            CONTENT REQUIREMENTS:
            - Use the research findings as your foundation
            - Make it engaging and accessible for {target_audience}
            - Include clear structure with headers and sections
            - Aim for at least 800-1200 words
            - Include practical insights and actionable information
            - Use examples and data from the research
            
            CONTENT STRUCTURE:
            1. Compelling headline/title
            2. Engaging introduction that hooks the reader
            3. Well-organized main content with clear sections
            4. Key insights and takeaways
            5. Conclusion with future outlook
            
            Make it informative yet engaging, professional yet accessible.""",
            agent=writer,
            expected_output=f"A well-structured, engaging {content_type} that transforms research into accessible content for {target_audience}.",
            context=[research_task]
        )

        # Create editing task
        editing_task = Task(
            description=f"""Edit and format the {content_type} to publication standards with excellent readability.
            
            EDITING CHECKLIST:
            1. Ensure clear, logical structure and flow
            2. Add proper headings, subheadings, and formatting
            3. Improve readability with bullet points and numbered lists
            4. Enhance engagement while maintaining professionalism
            5. Verify all key points from research are included
            6. Add a compelling title and meta description
            7. Ensure content is appropriate for {target_audience}
            8. Create a clean, professional final format
            
            FINAL FORMAT:
            # [Compelling Title]
            
            ## Executive Summary
            [Brief overview of key points]
            
            ## [Main sections with clear headers]
            [Well-formatted content with bullet points, examples, etc.]
            
            ## Key Takeaways
            [Bulleted list of main insights]
            
            ## Conclusion
            [Summary and future outlook]
            
            ---
            *Research completed: [Current date]*
            *Topic: {topic}*""",
            agent=editor,
            expected_output="A professionally formatted, publication-ready article with excellent structure and readability.",
            context=[research_task, writing_task]
        )

        # Create the crew
        crew = Crew(
            agents=[researcher, writer, editor],
            tasks=[research_task, writing_task, editing_task],
            process=Process.sequential,
            verbose=True
        )

        # Execute the crew's tasks
        print(f"\n🔍 Researching and writing about {topic}...\n")
        
        try:
            result = crew.kickoff()
            
            # Clean up the result
            clean_result = self._clean_content(str(result))
            
            # Format the final output
            formatted_result = self._format_research_output(clean_result, topic, content_type, target_audience)
            
            print("\n" + "="*80)
            print("🎉 RESEARCH & WRITING COMPLETE! 🎉")
            print("="*80)
            print(formatted_result)
            print("="*80)
            
            return {
                "topic": topic,
                "content_type": content_type,
                "target_audience": target_audience,
                "content": clean_result,
                "formatted_content": formatted_result
            }
            
        except Exception as e:
            print(f"\n❌ Error during research and writing: {e}")
            return {
                "topic": topic,
                "content_type": content_type,
                "target_audience": target_audience,
                "error": str(e)
            }
    
    def _clean_content(self, content: str) -> str:
        """Clean content of escape characters and formatting issues."""
        if not content:
            return ""
            
        clean_content = str(content)
        
        # Replace common escape characters
        clean_content = clean_content.replace('\\n', '\n')
        clean_content = clean_content.replace('\\r', '\r')
        clean_content = clean_content.replace('\\t', '\t')
        clean_content = clean_content.replace('\\"', '"')
        clean_content = clean_content.replace("\\'", "'")
        clean_content = clean_content.replace('\\\\', '\\')
        
        # Remove remaining escape sequences
        clean_content = re.sub(r'\\(.)', r'\1', clean_content)
        
        # Normalize whitespace
        clean_content = re.sub(r'\s+', ' ', clean_content)
        clean_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', clean_content)
        clean_content = clean_content.strip()
        
        return clean_content
    
    def _format_research_output(self, content: str, topic: str, content_type: str, target_audience: str) -> str:
        """Format the research output in a professional way."""
        
        # Clean the content first
        clean_content = self._clean_content(content)
        
        # Create header
        header = f"""📄 RESEARCH & WRITING COMPLETE 📄

Topic: {topic}
Content Type: {content_type}
Target Audience: {target_audience}
Generated: {os.popen('date').read().strip()}

{"-" * 80}

"""
        
        # Format content with proper markdown structure
        formatted_content = clean_content
        
        # Ensure proper markdown formatting
        if not formatted_content.startswith('#'):
            # Add a title if none exists
            lines = formatted_content.split('\n')
            title_line = f"# {topic}: Research and Analysis"
            formatted_content = title_line + '\n\n' + '\n'.join(lines)
        
        # Final result
        result = header + formatted_content
        return self._clean_content(result)

class LinkedInPostGenerator:
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the LinkedInPostGenerator with intelligent LLM selection."""
        self.llm, self.llm_type, self.model_used = setup_intelligent_llm(openai_api_key)
        print(f"🧠 Using {self.llm_type.upper()} with model: {self.model_used}")

    def _setup_openai_llm(self, openai_api_key: Optional[str] = None):
        """Set up OpenAI LLM for LinkedIn post creation."""
        
        print("🔄 Setting up OpenAI for LinkedIn Post creation...")
        
        # Set up API key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        elif not os.environ.get("OPENAI_API_KEY"):
            # No hardcoded key - require user to provide one
            raise Exception("OpenAI API key required. Please set OPENAI_API_KEY environment variable or pass --openai-api-key parameter.")
        
        try:
            from langchain_openai import ChatOpenAI
            
            llm = ChatOpenAI(
                model="gpt-4",
                temperature=0.8,  # Higher creativity for social media content
                max_tokens=1000
            )
            
            print("🧪 Testing OpenAI connection...")
            test_response = llm.invoke("Hello! Say 'Hi' back.")
            print(f"✅ OpenAI connection successful: {str(test_response)[:50]}...")
            return llm
                
        except Exception as e:
            print(f"❌ Failed to setup OpenAI: {e}")
            raise Exception("OpenAI setup failed. Please check your API key.")

    def create_linkedin_post(self, 
                           topic: str,
                           post_type: str = "professional insights",
                           tone: str = "engaging") -> Dict:
        """Create viral LinkedIn content using AI agents."""
        
        print(f"📱 Creating LinkedIn post about '{topic}'")
        print(f"📝 Post Type: {post_type} | 🎯 Tone: {tone}")
        print("🧠 Using OpenAI GPT-4 for viral content creation")

        # Create the career coach/researcher agent
        coach = Agent(
            role='Senior Career Coach & Trend Analyst',
            goal=f'Discover and examine key insights, trends, and valuable information about {topic}',
            backstory=f"""You are an expert in spotting emerging trends and essential insights in technology, AI, 
            business, and professional development. You have 15+ years of experience in career coaching and industry analysis.
            You excel at identifying what professionals need to know, current market trends, and actionable insights.
            Your expertise covers the latest developments in tech, AI, business strategies, and career advancement.
            You provide data-driven insights, statistics, and real-world examples that resonate with professionals.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

        # Create the LinkedIn influencer writer agent
        influencer = Agent(
            role='LinkedIn Influencer & Content Creator',
            goal=f'Write catchy, engaging LinkedIn posts that drive engagement and provide value',
            backstory=f"""You are a specialized LinkedIn content creator with millions of followers and viral posts.
            You understand exactly what makes content go viral on LinkedIn - the perfect mix of value, storytelling,
            and engagement tactics. You excel at writing in a {tone} tone and creating {post_type} content.
            Your posts always include strategic emoji usage, compelling hooks, clear value propositions,
            and calls-to-action that drive meaningful engagement. You know how to structure posts for maximum readability
            and impact, using line breaks, bullet points, and formatting that works perfectly on LinkedIn.""",
            verbose=True,
            allow_delegation=True,
            llm=self.llm
        )

        # Create the content critic/editor agent
        critic = Agent(
            role='Expert LinkedIn Content Strategist',
            goal='Refine and optimize LinkedIn posts for maximum engagement and viral potential',
            backstory=f"""You are a LinkedIn content strategist who has helped create hundreds of viral posts.
            You have deep expertise in LinkedIn algorithm optimization, engagement psychology, and content performance.
            You ensure posts are perfectly formatted, have compelling headlines, strategic emoji placement,
            optimal length (typically 1300-1500 characters for LinkedIn), proper hashtag usage, and strong calls-to-action.
            You focus on making content scannable, valuable, and shareable while maintaining authenticity and professionalism.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

        # Create research/insights task
        research_task = Task(
            description=f"""Research and compile key insights about {topic}.
            
            RESEARCH REQUIREMENTS:
            - Identify 3-5 key trends, insights, or actionable tips about {topic}
            - Find current statistics, data points, or interesting facts
            - Identify pain points or challenges professionals face
            - Discover emerging opportunities or solutions
            - Include specific examples or case studies if relevant
            
            DELIVERABLE:
            Create a research brief with:
            - Executive summary of key findings
            - 3-5 main insights or trends (in bullet points)
            - Supporting data, statistics, or examples
            - Professional challenges and opportunities
            - Actionable takeaways for LinkedIn audience
            
            Focus on insights that will resonate with professionals and provide genuine value.""",
            agent=coach,
            expected_output="A comprehensive research brief with key insights, trends, and actionable information formatted in bullet points."
        )

        # Create LinkedIn post writing task
        writing_task = Task(
            description=f"""Create an engaging LinkedIn post about {topic} using the research insights.
            
            POST REQUIREMENTS:
            - Write in a {tone} tone for {post_type} content
            - Use the research insights as your foundation
            - Create a compelling hook in the first line
            - Include 3-5 key insights or tips in an easy-to-scan format
            - Add strategic emojis for visual appeal (but don't overdo it)
            - Include relevant hashtags (3-5 maximum)
            - End with a strong call-to-action or engaging question
            - Keep it between 1200-1500 characters (optimal for LinkedIn)
            
            POST STRUCTURE:
            1. Compelling hook/opening line
            2. Brief context or personal insight
            3. Main content (insights/tips) in scannable format
            4. Key takeaway or conclusion
            5. Call-to-action or engagement question
            6. Relevant hashtags
            
            Make it valuable, authentic, and shareable. Focus on providing real value to professionals.""",
            agent=influencer,
            expected_output=f"An engaging, well-structured LinkedIn post about {topic} with strategic formatting and clear value proposition.",
            context=[research_task]
        )

        # Create content optimization task
        optimization_task = Task(
            description=f"""Optimize and refine the LinkedIn post for maximum engagement and viral potential.
            
            OPTIMIZATION CHECKLIST:
            1. Ensure the hook grabs attention in the first 2 lines
            2. Verify optimal length (1200-1500 characters)
            3. Improve readability with proper line breaks and spacing
            4. Optimize emoji usage (strategic, not excessive)
            5. Enhance call-to-action for better engagement
            6. Verify hashtag relevance and quantity (3-5 max)
            7. Ensure value proposition is crystal clear
            8. Check for LinkedIn algorithm optimization
            
            FINAL FORMAT:
            [Compelling hook - first 1-2 lines]
            
            [Main content with proper spacing and formatting]
            
            [Strong call-to-action or question]
            
            [3-5 relevant hashtags]
            
            ---
            Character count: [X/1500]
            
            Make it scroll-stopping, valuable, and perfectly formatted for LinkedIn success.""",
            agent=critic,
            expected_output="A perfectly optimized LinkedIn post ready for publishing with character count and formatting notes.",
            context=[research_task, writing_task]
        )

        # Create the crew
        crew = Crew(
            agents=[coach, influencer, critic],
            tasks=[research_task, writing_task, optimization_task],
            process=Process.sequential,
            verbose=True
        )

        # Execute the crew's tasks
        print(f"\n📱 Creating viral LinkedIn content about {topic}...\n")
        
        try:
            result = crew.kickoff()
            
            # Clean up the result
            clean_result = self._clean_content(str(result))
            
            # Format the final output
            formatted_result = self._format_linkedin_output(clean_result, topic, post_type, tone)
            
            print("\n" + "="*80)
            print("🚀 LINKEDIN POST READY! 🚀")
            print("="*80)
            print(formatted_result)
            print("="*80)
            
            return {
                "topic": topic,
                "post_type": post_type,
                "tone": tone,
                "content": clean_result,
                "formatted_content": formatted_result
            }
            
        except Exception as e:
            print(f"\n❌ Error during LinkedIn post creation: {e}")
            return {
                "topic": topic,
                "post_type": post_type,
                "tone": tone,
                "error": str(e)
            }
    
    def _clean_content(self, content: str) -> str:
        """Clean content of escape characters and formatting issues."""
        if not content:
            return ""
            
        clean_content = str(content)
        
        # Replace common escape characters
        clean_content = clean_content.replace('\\n', '\n')
        clean_content = clean_content.replace('\\r', '\r')
        clean_content = clean_content.replace('\\t', '\t')
        clean_content = clean_content.replace('\\"', '"')
        clean_content = clean_content.replace("\\'", "'")
        clean_content = clean_content.replace('\\\\', '\\')
        
        # Remove remaining escape sequences
        clean_content = re.sub(r'\\(.)', r'\1', clean_content)
        
        # Normalize whitespace but preserve intentional line breaks
        clean_content = re.sub(r' +', ' ', clean_content)  # Multiple spaces to single
        clean_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', clean_content)  # Multiple newlines to double
        clean_content = clean_content.strip()
        
        return clean_content
    
    def _format_linkedin_output(self, content: str, topic: str, post_type: str, tone: str) -> str:
        """Format the LinkedIn output for easy copying and posting."""
        
        # Clean the content first
        clean_content = self._clean_content(content)
        
        # Create header
        header = f"""📱 LINKEDIN POST READY 📱

Topic: {topic}
Type: {post_type}
Tone: {tone}
Generated: {os.popen('date').read().strip()}

{"-" * 60}
COPY & PASTE TO LINKEDIN:
{"-" * 60}

"""
        
        # Extract just the final post content (usually the last task result)
        lines = clean_content.split('\n')
        
        # Find the actual post content (look for the optimized version)
        post_content = clean_content
        if "Character count:" in clean_content:
            # Extract everything before the character count line
            post_lines = []
            for line in lines:
                if "Character count:" in line or "---" in line:
                    break
                post_lines.append(line)
            post_content = '\n'.join(post_lines).strip()
        
        # Final result with helpful footer
        footer = f"""

{"-" * 60}
💡 LinkedIn Posting Tips:
• Post during peak hours (8-10am, 12-2pm, 5-6pm)
• Engage with comments within the first hour
• Tag relevant people or companies if appropriate
• Consider posting as a carousel for longer content
{"-" * 60}"""
        
        result = header + post_content + footer
        return self._clean_content(result)

class TravelPlannerGenerator:
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the TravelPlannerGenerator with intelligent LLM selection."""
        self.llm, self.llm_type, self.model_used = setup_intelligent_llm(openai_api_key)
        print(f"🧠 Using {self.llm_type.upper()} with model: {self.model_used}")

    def create_travel_plan(self,
                          destination: str,
                          duration: str,
                          budget: str,
                          preferences: str = "balanced") -> str:
        """Generate a comprehensive travel plan using CrewAI agents."""
        print(f"✈️ Creating travel plan for {destination}")
        print(f"🔄 Setting up {self.llm_type.upper()} for travel planning...")
        if self.llm_type == 'openai':
            print("🔑 Using default OpenAI API key")
        else:
            print(f"🦙 Using Ollama model: {self.model_used}")
        
        # Create specialized agents for travel planning
        destination_researcher = Agent(
            role="Destination Research Specialist",
            goal=f"Research comprehensive information about {destination} including attractions, culture, and logistics",
            backstory="""You are an experienced travel researcher who knows how to find the best 
            attractions, hidden gems, cultural insights, and practical travel information for any destination.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        budget_planner = Agent(
            role="Budget Planning Expert",
            goal=f"Create a detailed budget breakdown for {duration} in {destination} within {budget} budget",
            backstory="""You are a financial planning expert specializing in travel budgets. You excel 
            at finding cost-effective options while maximizing value and experience.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        itinerary_creator = Agent(
            role="Itinerary Planning Specialist", 
            goal=f"Create a day-by-day itinerary for {duration} in {destination} matching {preferences} preferences",
            backstory="""You are a master trip planner who creates perfectly timed itineraries that 
            balance must-see attractions, local experiences, rest time, and logistics.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # Create tasks for each agent
        research_task = Task(
            description=f"""Research {destination} thoroughly including:
            - Top attractions and hidden gems
            - Local culture and customs
            - Best time to visit specific places
            - Transportation options
            - Local cuisine recommendations
            - Safety considerations""",
            agent=destination_researcher,
            expected_output="Comprehensive destination research with attractions, culture, and logistics"
        )
        
        budget_task = Task(
            description=f"""Create a detailed budget plan for {duration} in {destination} with {budget} budget:
            - Accommodation options and costs
            - Transportation costs
            - Food and dining budget
            - Activities and entrance fees
            - Shopping and miscellaneous expenses
            - Money-saving tips""",
            agent=budget_planner,
            expected_output="Detailed budget breakdown with cost estimates and money-saving tips"
        )
        
        itinerary_task = Task(
            description=f"""Create a detailed day-by-day itinerary for {duration} in {destination} that:
            - Matches {preferences} travel style
            - Fits within the budget
            - Includes the best researched attractions
            - Balances activities with rest time
            - Considers logistics and transportation
            - Includes meal suggestions""",
            agent=itinerary_creator,
            expected_output="Complete day-by-day itinerary with timings, activities, and practical details",
            context=[research_task, budget_task]
        )
        
        # Create and run the crew
        travel_crew = Crew(
            agents=[destination_researcher, budget_planner, itinerary_creator],
            tasks=[research_task, budget_task, itinerary_task],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            result = travel_crew.kickoff()
            return self._format_travel_output(str(result), destination, duration, budget, preferences)
        except Exception as e:
            error_msg = f"Error creating travel plan: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg

    def _clean_content(self, content: str) -> str:
        """Clean and format the content."""
        # Remove any markdown code blocks
        content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
        
        # Remove any remaining backticks
        content = content.replace('`', '')
        
        # Clean up extra whitespace
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = content.strip()
        
        return content

    def _format_travel_output(self, content: str, destination: str, duration: str, budget: str, preferences: str) -> str:
        """Format the travel plan output."""
        content = self._clean_content(content)
        
        # Add a header
        header = f"""
🌍 Travel Plan for {destination}
⏱️ Duration: {duration}
💰 Budget: {budget}
🎯 Style: {preferences}

"""
        
        # Add a footer
        footer = f"""

---
💡 Tips:
• Book accommodations in advance
• Check local weather forecasts
• Get travel insurance
• Keep digital copies of important documents
• Learn basic local phrases
• Download offline maps
• Register with your embassy if traveling internationally

Happy travels! 🚀
"""
        
        return header + content + footer 


class InstagramPostGenerator:
    """Instagram Post Generator using CrewAI"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.llm, self.llm_type, self.model_used = setup_intelligent_llm(openai_api_key)
    
    def create_instagram_post(self, 
                            topic: str,
                            target_audience: str = "general audience",
                            post_type: str = "engaging",
                            include_hashtags: bool = True) -> Dict:
        """
        Create Instagram post content with strategy, copy, and hashtag recommendations
        
        Args:
            topic: Main topic for the Instagram post
            target_audience: Target audience for the post
            post_type: Type of post (engaging, educational, promotional, inspirational)
            include_hashtags: Whether to include hashtag recommendations
            
        Returns:
            Dict containing the generated content and metadata
        """
        try:
            print(f"🎯 Creating Instagram post about: {topic}")
            print(f"👥 Target audience: {target_audience}")
            print(f"📱 Post type: {post_type}")
            print(f"🔧 Using {self.llm_type} with model: {self.model_used}")
            
            # Market Research Agent
            market_researcher = Agent(
                role="Instagram Market Researcher",
                goal=f"Research trending topics, hashtags, and engagement strategies related to {topic} for Instagram",
                backstory="""You are an expert Instagram market researcher with deep knowledge of social media trends, 
                hashtag performance, and audience engagement patterns. You excel at identifying viral content opportunities 
                and understanding what makes posts successful on Instagram. You stay up-to-date with the latest Instagram 
                algorithm changes and best practices for maximum reach and engagement.""",
                verbose=True,
                llm=self.llm,
                allow_delegation=False
            )
            
            # Content Strategist Agent
            content_strategist = Agent(
                role="Instagram Content Strategist",
                goal=f"Develop a comprehensive content strategy for an Instagram post about {topic} targeting {target_audience}",
                backstory="""You are a seasoned Instagram content strategist who understands the platform's unique 
                characteristics and audience behaviors. You excel at creating content strategies that balance 
                entertainment, education, and engagement. You know how to craft posts that not only look good 
                but also drive meaningful interactions and build community.""",
                verbose=True,
                llm=self.llm,
                allow_delegation=False
            )
            
            # Copywriter Agent
            copywriter = Agent(
                role="Instagram Copywriter",
                goal=f"Write compelling, engaging Instagram post copy about {topic} that resonates with {target_audience}",
                backstory="""You are a creative Instagram copywriter who knows how to craft posts that stop the scroll. 
                You understand the art of writing for social media - keeping it concise yet impactful, using the right 
                tone for the audience, and including clear calls-to-action. You know how to use emojis effectively 
                and create copy that encourages engagement through comments, likes, and shares.""",
                verbose=True,
                llm=self.llm,
                allow_delegation=False
            )
            
            # Hashtag Specialist Agent
            hashtag_specialist = Agent(
                role="Instagram Hashtag Specialist",
                goal=f"Research and recommend the most effective hashtags for a post about {topic}",
                backstory="""You are an Instagram hashtag expert who understands the science behind hashtag strategy. 
                You know how to mix popular, niche, and branded hashtags to maximize reach while targeting the right 
                audience. You stay updated on trending hashtags and understand which ones are overused or banned. 
                You create hashtag sets that help posts get discovered by the ideal audience.""",
                verbose=True,
                llm=self.llm,
                allow_delegation=False
            )
            
            # Define tasks
            market_research_task = Task(
                description=f"""Conduct comprehensive market research for an Instagram post about {topic}. 
                Research current trends, popular content formats, optimal posting times, and audience preferences. 
                Analyze what type of content performs well in this niche and identify opportunities for engagement.
                
                Focus on:
                - Current trending topics related to {topic}
                - Popular content formats (carousel, video, single image, etc.)
                - Audience engagement patterns
                - Competitor analysis in this space
                - Best posting times for {target_audience}
                
                Provide actionable insights that will inform the content strategy.""",
                agent=market_researcher,
                expected_output="A detailed market research report with trending topics, audience insights, and content format recommendations"
            )
            
            content_strategy_task = Task(
                description=f"""Based on the market research, develop a comprehensive content strategy for an Instagram post about {topic}.
                The strategy should be tailored for {target_audience} and focus on creating {post_type} content.
                
                Include:
                - Content angle and key messaging
                - Visual content recommendations
                - Engagement strategy (how to encourage comments, shares, saves)
                - Call-to-action suggestions
                - Post format recommendation (single image, carousel, reel, etc.)
                - Optimal posting time
                
                Make sure the strategy aligns with current Instagram best practices and trends.""",
                agent=content_strategist,
                expected_output="A comprehensive content strategy document with specific recommendations for the Instagram post",
                context=[market_research_task]
            )
            
            copywriting_task = Task(
                description=f"""You are an expert Instagram copywriter. Write compelling Instagram post copy about {topic} for {target_audience}.
                
                Create {post_type} content that includes:
                
                1. A HOOK (first line that grabs attention)
                2. MAIN CONTENT (2-3 sentences about {topic})
                3. CALL TO ACTION (asking for engagement)
                4. EMOJIS (naturally integrated)
                
                Example format:
                "🚀 Hook about {topic}...
                
                Main content explaining the value...
                
                What's your experience with {topic}? Share below! 👇
                
                #hashtag #hashtag"
                
                Make it authentic, engaging, and optimized for {target_audience}. 
                Keep it under 2200 characters. Focus ONLY on writing the actual post copy.""",
                agent=copywriter,
                expected_output="Ready-to-post Instagram copy with hook, content, call-to-action, and emojis",
                context=[market_research_task, content_strategy_task]
            )
            
            # Conditionally add hashtag task
            tasks = [market_research_task, content_strategy_task, copywriting_task]
            
            if include_hashtags:
                hashtag_task = Task(
                    description=f"""You are an Instagram hashtag expert. Research and provide EXACTLY 25 hashtags for a post about {topic} targeting {target_audience}.
                    
                    Provide hashtags in this format:
                    
                    HIGH REACH (5 hashtags - 100K+ posts):
                    #hashtag1 #hashtag2 #hashtag3 #hashtag4 #hashtag5
                    
                    MEDIUM REACH (10 hashtags - 10K-100K posts):
                    #hashtag6 #hashtag7 #hashtag8 #hashtag9 #hashtag10 #hashtag11 #hashtag12 #hashtag13 #hashtag14 #hashtag15
                    
                    LOW REACH (10 hashtags - Under 10K posts):
                    #hashtag16 #hashtag17 #hashtag18 #hashtag19 #hashtag20 #hashtag21 #hashtag22 #hashtag23 #hashtag24 #hashtag25
                    
                    Focus ONLY on providing the hashtags in the exact format above. Make them relevant to {topic} and {target_audience}.""",
                    agent=hashtag_specialist,
                    expected_output="25 Instagram hashtags categorized by reach potential in the specified format",
                    context=[market_research_task, content_strategy_task]
                )
                tasks.append(hashtag_task)
            
            # Create and run crew
            agents = [market_researcher, content_strategist, copywriter]
            if include_hashtags:
                agents.append(hashtag_specialist)
            
            crew = Crew(
                agents=agents,
                tasks=tasks,
                verbose=True,
                process=Process.sequential
            )
            
            print("🚀 Starting Instagram post creation process...")
            result = crew.kickoff()
            
            # Format the output
            formatted_result = self._format_instagram_output(
                result, topic, target_audience, post_type, include_hashtags
            )
            
            return {
                'content': str(result),
                'formatted_content': formatted_result,
                'topic': topic,
                'target_audience': target_audience,
                'post_type': post_type,
                'llm_used': f"{self.llm_type}:{self.model_used}",
                'include_hashtags': include_hashtags
            }
            
        except Exception as e:
            error_msg = f"Error creating Instagram post: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                'content': error_msg,
                'formatted_content': error_msg,
                'topic': topic,
                'target_audience': target_audience,
                'post_type': post_type,
                'error': str(e)
            }
    
    def _format_instagram_output(self, content: str, topic: str, target_audience: str, post_type: str, include_hashtags: bool) -> str:
        """Format the Instagram post output for better readability"""
        
        # Clean the content
        cleaned_content = self._clean_content(str(content))
        
        # Create formatted output
        formatted_output = f"""
# 📱 INSTAGRAM POST STRATEGY

## 🎯 Post Details
- **Topic**: {topic}
- **Target Audience**: {target_audience}
- **Post Type**: {post_type}
- **Hashtags Included**: {'Yes' if include_hashtags else 'No'}

## 📋 Generated Content

{cleaned_content}

## 💡 Next Steps
1. Review and customize the copy to match your brand voice
2. Create or source visual content based on the recommendations
3. Schedule the post for optimal engagement times
4. Monitor performance and engage with comments
5. Consider creating variations for Stories or Reels

---
*Generated by IdeaWeaver Instagram Post Agent*
"""
        
        return formatted_output.strip()
    
    def _clean_content(self, content: str) -> str:
        """Clean and format the content"""
        # Remove <think> tags and their content (from reasoning models)
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        
        # Remove any markdown code blocks
        content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
        
        # Remove any remaining backticks
        content = content.replace('`', '')
        
        # Remove "Begin! This is VERY important..." prompts
        content = re.sub(r'Begin! This is VERY important.*?depends on it!', '', content, flags=re.DOTALL)
        
        # Remove "Thought:" lines
        content = re.sub(r'Thought:.*?\n', '', content)
        
        # Clean up extra whitespace
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = content.strip()
        
        return content


class StockAnalysisGenerator:
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the StockAnalysisGenerator with intelligent LLM selection."""
        self.llm, self.llm_type, self.model_used = setup_intelligent_llm(openai_api_key)
        print(f"🧠 Using {self.llm_type.upper()} with model: {self.model_used}")

    def analyze_stock(self, symbol: str) -> str:
        """Generate comprehensive stock analysis using CrewAI agents."""
        print(f"📈 Analyzing stock: {symbol}")
        print(f"🧠 Checking OpenAI GPT-4 availability for stock analysis")
        if self.llm_type == 'openai':
            print("🔑 Using default OpenAI API key")
        else:
            print(f"🦙 Using Ollama model: {self.model_used}")
        
        # Create specialized agents for stock analysis
        news_researcher = Agent(
            role="Stock News Researcher",
            goal=f"Research the latest news, press releases, and market sentiment for {symbol}",
            backstory="""You are an expert financial news researcher with access to real-time market data. 
            You excel at finding relevant news, analyzing market sentiment, and identifying key events that 
            could impact stock performance.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        financial_analyst = Agent(
            role="Financial Analyst", 
            goal=f"Analyze the financial health and performance of {symbol}",
            backstory="""You are a seasoned financial analyst with expertise in fundamental analysis, 
            financial statements, valuation models, and industry comparisons. You provide detailed 
            insights into company performance and financial metrics.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        investment_advisor = Agent(
            role="Investment Advisor",
            goal=f"Provide investment recommendation for {symbol} based on comprehensive analysis",
            backstory="""You are a professional investment advisor who synthesizes research and analysis 
            to provide clear, actionable investment recommendations. You consider risk factors, market 
            conditions, and investment timeframes.""",
            verbose=True, 
            allow_delegation=False,
            llm=self.llm
        )
        
        # Create tasks for each agent
        news_research_task = Task(
            description=f"""Research the latest news, press releases, and market sentiment for {symbol}. Focus on:
            - Recent headlines
            - Major events
            - Analyst opinions
            - Regulatory changes
            - Market sentiment""",
            agent=news_researcher,
            expected_output="Detailed summary of recent news and market sentiment"
        )
        
        financial_analysis_task = Task(
            description=f"""Analyze the financial health and performance of {symbol}. Consider:
            - Revenue, profit, and growth trends
            - Valuation metrics (P/E, P/S, etc.)
            - Recent earnings reports
            - Industry comparisons
            - Risks and opportunities""",
            agent=financial_analyst,
            expected_output="Comprehensive financial analysis with key metrics and trends"
        )
        
        investment_recommendation_task = Task(
            description=f"""Based on the news and financial analysis, summarize the outlook for {symbol} and provide a clear recommendation (buy/hold/sell) with reasoning.""",
            agent=investment_advisor,
            expected_output="Clear investment recommendation with supporting rationale",
            context=[news_research_task, financial_analysis_task]
        )
        
        # Create and run the crew
        analysis_crew = Crew(
            agents=[news_researcher, financial_analyst, investment_advisor],
            tasks=[news_research_task, financial_analysis_task, investment_recommendation_task],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            result = analysis_crew.kickoff()
            return self._format_stock_output(str(result), symbol)
        except Exception as e:
            error_msg = f"Error during stock analysis: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg

    def _format_stock_output(self, content: str, symbol: str) -> str:
        header = f"""
📈 Stock Analysis for {symbol}
{'='*60}
"""
        footer = f"""
{'='*60}
⚠️ This analysis is for informational purposes only and not financial advice.
"""
        return header + str(content).strip() + footer 

class SystemDiagnosticGenerator:
    """
    System Diagnostic Generator using CrewAI agents
    
    This class creates two agents:
    1. System Diagnostic Agent - executes actual Linux/macOS system diagnostic commands
    2. System Advisory Agent - provides recommendations based on real diagnostic output
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the System Diagnostic Generator"""
        self.llm, self.llm_type, self.model_name = setup_intelligent_llm(openai_api_key)
        print(f"🔧 System Diagnostic Generator initialized with {self.llm_type} ({self.model_name})")
    
    def _execute_system_commands(self) -> Dict[str, str]:
        """Execute actual system diagnostic commands and return their output"""
        import subprocess
        import platform
        
        results = {}
        system = platform.system()
        
        try:
            # 1. CPU & Load Analysis - 'w' command (works on both Linux and macOS)
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
            
            # 2. Memory Analysis - different commands for different systems
            print("🧠 Executing Memory analysis...")
            if system == "Darwin":  # macOS
                try:
                    # Use vm_stat for macOS
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
            
            # 3. Network Analysis - use ip or ifconfig
            print("🌐 Executing Network analysis...")
            network_commands = []
            if system == "Darwin":  # macOS
                network_commands = [
                    (['ifconfig'], 'ifconfig'),
                    (['netstat', '-rn'], 'netstat -rn'),  # routing table
                    (['netstat', '-an'], 'netstat -an')   # active connections
                ]
            else:  # Linux
                network_commands = [
                    (['ip', 'addr', 'show'], 'ip addr show'),
                    (['ip', 'route', 'show'], 'ip route show'),
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
                'command': 'Network analysis commands',
                'output': '\n\n'.join(network_output),
                'success': len(network_output) > 0
            }
            
            # 4. Process & I/O Analysis - top command
            print("⚡ Executing Process & I/O analysis...")
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
            
            # 5. Additional macOS-specific commands
            if system == "Darwin":
                print("🍎 Executing macOS-specific diagnostics...")
                try:
                    # Get CPU core count
                    result = subprocess.run(['sysctl', '-n', 'hw.ncpu'], capture_output=True, text=True, timeout=5)
                    cpu_cores = result.stdout.strip() if result.returncode == 0 else "unknown"
                    
                    # Get system uptime and load with CPU context
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
    
    def run_system_diagnostics(self, verbose: bool = False) -> Dict:
        """
        Run comprehensive system diagnostics and provide recommendations
        
        Returns:
            Dict containing diagnostic results and recommendations
        """
        try:
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
                goal="Analyze REAL system diagnostic output and provide actionable recommendations for optimization",
                backstory="""You are a senior systems performance consultant with decades of experience 
                in both Linux and macOS system optimization. You excel in analyzing system metrics from 
                diagnostic commands and translating technical data into clear, actionable recommendations. 
                You understand the relationships between CPU load, memory usage, network performance, and 
                I/O operations, and can identify performance bottlenecks and suggest specific solutions.
                
                IMPORTANT: You are analyzing REAL system output, not simulated data. Pay close attention 
                to the actual values and system state provided.""",
                verbose=verbose,
                llm=self.llm,
                allow_delegation=False
            )
            
            # Create advisory task with real diagnostic data
            advisory_task = Task(
                description=f"""Analyze the following REAL system diagnostic output and provide comprehensive 
recommendations for system optimization and performance improvement.

ACTUAL SYSTEM DIAGNOSTIC DATA:
{diagnostic_data}

Based on this REAL diagnostic data, provide:

1. **Overall System Health Assessment**: Evaluate the current system state based on actual metrics.

2. **CPU & Load Analysis**: 
   - Interpret the actual load averages shown
   - Consider the number of CPU cores when assessing load
   - Identify if the system is appropriately loaded or has issues

3. **Memory Analysis**:
   - Analyze the real memory usage data
   - Identify any memory pressure or issues
   - Provide specific recommendations based on actual usage

4. **Network Analysis**: 
   - Review actual network interface data
   - Identify any network-related issues from real output
   - Check actual network connections and ports

5. **Process Analysis**:
   - Identify actual processes consuming resources
   - Analyze real I/O and CPU usage patterns
   - Recommend specific actions based on actual process data

6. **Actionable Recommendations**:
   - Provide specific steps based on the actual system state
   - Include real command examples where appropriate
   - Focus on addressing issues found in the real data

7. **Priority Assessment**: Rank recommendations by actual urgency based on real metrics

IMPORTANT: Base all analysis on the actual command output provided, not on assumptions or generic advice.""",
                agent=advisory_agent,
                expected_output="""A comprehensive system analysis report based on REAL system data with:
1. Executive Summary of actual system health
2. Detailed analysis of each diagnostic area based on real metrics
3. Prioritized list of actionable recommendations with specific commands
4. Performance optimization suggestions based on actual system state
5. Monitoring recommendations relevant to the current system

Format with clear headers, bullet points, and priority indicators."""
            )
            
            # Create and execute crew with just the advisory agent
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
                'diagnostic_output': str(result),
                'raw_commands': command_results,
                'llm_used': f"{self.llm_type} ({self.model_name})",
                'agents_used': ['System Performance Advisor']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'llm_used': f"{self.llm_type} ({self.model_name})"
            }
    
    def _format_diagnostic_output(self, content: str) -> str:
        """Format the diagnostic output for better readability"""
        # Clean up the content
        clean_content = str(content)
        clean_content = clean_content.replace('\\n', '\n')
        clean_content = clean_content.replace('\\"', '"')
        clean_content = clean_content.replace("\\'", "'")
        
        # Add proper formatting
        clean_content = clean_content.replace('CPU & Load Analysis:', '\n🖥️  CPU & Load Analysis:\n' + '='*50)
        clean_content = clean_content.replace('Memory Analysis:', '\n🧠 Memory Analysis:\n' + '='*50)
        clean_content = clean_content.replace('Network Analysis:', '\n🌐 Network Analysis:\n' + '='*50)
        clean_content = clean_content.replace('Process Analysis:', '\n⚡ Process Analysis:\n' + '='*50)
        clean_content = clean_content.replace('Recommendations:', '\n💡 System Recommendations:\n' + '='*50)
        
        # Clean up excessive line breaks
        import re
        clean_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', clean_content)
        
        return clean_content 