#!/usr/bin/env python3
"""
Minimal CrewAI CLI - No heavy dependencies
"""
import click
import os
import sys
from pathlib import Path
import re
from .crew_ai import TravelPlannerGenerator, StockAnalysisGenerator

@click.group()
def crew():
    """CrewAI powered content generation and planning commands."""
    pass

@crew.command('generate_storybook')
@click.option('--theme', required=True, help='Theme of the storybook')
@click.option('--target-age', required=True, help='Target age group (e.g., "3-5", "6-8", "9-12")')
@click.option('--num-pages', default=1, help='Number of pages in the storybook')
@click.option('--style', default='whimsical', help='Writing style (e.g., "whimsical", "educational", "adventure")')
@click.option('--openai-api-key', envvar='OPENAI_API_KEY', help='OpenAI API key')
def generate_storybook(theme, target_age, num_pages, style, openai_api_key):
    """Generate a storybook using CrewAI agents.
    
    \b
    Examples:
      ideaweaver crew generate_storybook --theme "brave little mouse" --target-age "3-5"
      ideaweaver crew generate_storybook --theme "space adventure" --target-age "6-8" --num-pages 3 --style "adventure"
      ideaweaver crew generate_storybook --theme "learning colors" --target-age "2-4" --style "educational"
    """
    
    click.echo(f"🚀 Generating storybook: '{theme}' for ages {target_age}")
    click.echo(f"🧠 Using OpenAI GPT-4 (Ollama has compatibility issues with CrewAI)")
    
    try:
        # Import only when needed
        from .crew_ai import StorybookGenerator
        
        generator = StorybookGenerator()
        result = generator.create_storybook(
            theme=theme,
            target_age=target_age,
            num_pages=num_pages,
            style=style
        )
        
        # Extract just the clean content, no complex formatting
        if 'content' in result:
            clean_content = str(result['content'])
            # Simple cleaning
            clean_content = clean_content.replace('\\n', '\n')
            clean_content = clean_content.replace('\\"', '"')
            clean_content = clean_content.replace("\\'", "'")
            
            # Add proper line breaks for readability
            clean_content = clean_content.replace('Theme:', '\n\nTheme:')
            clean_content = clean_content.replace('Page 1:', '\n\nPage 1:')
            clean_content = clean_content.replace('Illustration:', '\n\nIllustration:')
            clean_content = clean_content.replace('Summary:', '\n\nSummary:')
            clean_content = clean_content.replace('. ', '.\n')
            
            # Clean up excessive line breaks
            clean_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', clean_content)
            
            click.echo("\n" + "="*60)
            click.echo(f"📚 STORYBOOK: {theme}")
            click.echo(f"👶 Age: {target_age} | 📄 Pages: {num_pages} | 🎨 Style: {style}")
            click.echo("="*60)
            click.echo(clean_content)
            click.echo("="*60)
        else:
            click.echo("❌ Error generating storybook")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)

@crew.command('research_write')
@click.option('--topic', required=True, help='Research topic to investigate and write about')
@click.option('--content-type', default='blog post', help='Type of content (blog post, article, report)')
@click.option('--audience', default='tech enthusiasts', help='Target audience for the content')
@click.option('--openai-api-key', envvar='OPENAI_API_KEY', help='OpenAI API key')
def research_write(topic, content_type, audience, openai_api_key):
    """Research a topic and generate well-formatted content using CrewAI agents.
    
    \b
    Examples:
      ideaweaver crew research_write --topic "AI in healthcare"
      ideaweaver crew research_write --topic "blockchain technology" --content-type "article" --audience "investors"
      ideaweaver crew research_write --topic "climate change solutions" --content-type "report" --audience "policymakers"
    """
    
    click.echo(f"🔍 Researching and writing about: '{topic}'")
    click.echo(f"📝 Content Type: {content_type} | 👥 Audience: {audience}")
    click.echo(f"🧠 Using OpenAI GPT-4 with web search capabilities")
    
    try:
        # Import only when needed
        from .crew_ai import ResearchWriterGenerator
        
        generator = ResearchWriterGenerator()
        result = generator.create_research_content(
            topic=topic,
            content_type=content_type,
            target_audience=audience
        )
        
        # Extract and display the content
        if 'content' in result:
            clean_content = str(result['content'])
            # Simple cleaning
            clean_content = clean_content.replace('\\n', '\n')
            clean_content = clean_content.replace('\\"', '"')
            clean_content = clean_content.replace("\\'", "'")
            
            # Clean up excessive line breaks
            clean_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', clean_content)
            
            click.echo("\n" + "="*80)
            click.echo(f"📄 RESEARCH & WRITING: {topic}")
            click.echo(f"📝 Type: {content_type} | 👥 Audience: {audience}")
            click.echo("="*80)
            click.echo(clean_content)
            click.echo("="*80)
        else:
            click.echo("❌ Error generating research content")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)

@crew.command('travel_plan')
@click.option('--destination', required=True, help='Travel destination')
@click.option('--duration', required=True, help='Trip duration (e.g., "5 days", "2 weeks")')
@click.option('--budget', required=True, help='Budget range (e.g., "$1000-2000", "luxury")')
@click.option('--preferences', default='balanced', help='Travel style preferences (e.g., "adventure", "relaxed", "balanced")')
@click.option('--openai-api-key', envvar='OPENAI_API_KEY', help='OpenAI API key')
def travel_plan(destination, duration, budget, preferences, openai_api_key):
    """Generate a detailed travel plan using CrewAI agents.
    
    \b
    Examples:
      ideaweaver crew travel_plan --destination "Tokyo" --duration "7 days" --budget "$2000-3000"
      ideaweaver crew travel_plan --destination "Paris" --duration "5 days" --budget "$1500" --preferences "romantic"
      ideaweaver crew travel_plan --destination "Bali" --duration "10 days" --budget "luxury" --preferences "relaxed"
    """
    try:
        click.echo(f"✈️ Creating travel plan for: {destination}")
        click.echo(f"⏱️ Duration: {duration} | 💰 Budget: {budget} | 🎯 Style: {preferences}")
        
        # Initialize the travel planner
        click.echo("🧠 Using OpenAI GPT-4 for travel planning")
        
        planner = TravelPlannerGenerator(openai_api_key=openai_api_key)
        result = planner.create_travel_plan(
            destination=destination,
            duration=duration,
            budget=budget,
            preferences=preferences
        )
        
        # Display the travel plan
        click.echo("\n📋 Your Travel Plan:")
        click.echo("=" * 80)
        click.echo(result)
        click.echo("=" * 80)
        
    except Exception as e:
        click.echo(f"❌ Error creating travel plan: {str(e)}")
        raise click.ClickException(str(e))

@crew.command('stock_analysis')
@click.option('--symbol', required=True, help='Stock ticker symbol (e.g., "AAPL")')
@click.option('--openai-api-key', envvar='OPENAI_API_KEY', help='OpenAI API key')
def stock_analysis(symbol, openai_api_key):
    """Analyze a stock using CrewAI agents.
    
    \b
    Examples:
      ideaweaver crew stock_analysis --symbol AAPL
      ideaweaver crew stock_analysis --symbol TSLA
      ideaweaver crew stock_analysis --symbol GOOGL
    """
    try:
        click.echo(f"📈 Analyzing stock: {symbol}")
        generator = StockAnalysisGenerator(openai_api_key=openai_api_key)
        result = generator.analyze_stock(symbol)
        click.echo("\n" + result)
    except Exception as e:
        click.echo(f"❌ Error during stock analysis: {str(e)}")
        raise click.ClickException(str(e))

@click.command()
def check_llm_status():
    """Check the status of available LLM providers (Ollama and OpenAI).
    
    \b
    Examples:
      ideaweaver crew check_llm_status
      ./ideaweaver.sh crew check_llm_status
    """
    print("🔍 Checking LLM Status...\n")
    
    # Check Ollama
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            models_data = response.json()
            available_models = [model['name'] for model in models_data.get('models', [])]
            
            if available_models:
                print("✅ Ollama Status: AVAILABLE")
                print(f"📋 Available models ({len(available_models)}):")
                for model in available_models:
                    print(f"   🦙 {model}")
            else:
                print("⚠️  Ollama Status: RUNNING but no models installed")
                print("💡 Install a model: ollama pull deepseek-r1:1.5b")
        else:
            print("❌ Ollama Status: NOT RESPONDING")
            
    except requests.exceptions.RequestException:
        print("❌ Ollama Status: NOT RUNNING")
        print("💡 Start Ollama: ollama serve")
    except Exception as e:
        print(f"❌ Ollama Status: ERROR - {e}")
    
    print()
    
    # Check OpenAI
    try:
        import os
        if os.environ.get("OPENAI_API_KEY"):
            print("✅ OpenAI Status: API KEY CONFIGURED")
        else:
            print("⚠️  OpenAI Status: NO API KEY SET")
            print("💡 Set API key: export OPENAI_API_KEY=your_key")
    except Exception as e:
        print(f"❌ OpenAI Status: ERROR - {e}")
    
    print("\n🎯 Priority: Ollama (free) → OpenAI (paid fallback)")
    
    # Show intelligent setup result
    try:
        from .crew_ai import setup_intelligent_llm
        print("\n🧠 Testing intelligent LLM setup...")
        llm, llm_type, model_used = setup_intelligent_llm()
        print(f"✅ Result: Using {llm_type.upper()} with {model_used}")
    except Exception as e:
        print(f"❌ LLM Setup Error: {e}")

# Register commands with the crew group
crew.add_command(travel_plan)
crew.add_command(stock_analysis)
crew.add_command(check_llm_status)

if __name__ == '__main__':
    crew() 