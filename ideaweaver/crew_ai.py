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