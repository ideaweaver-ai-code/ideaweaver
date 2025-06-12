@click.group(help="""Langfuse monitoring commands for IdeaWeaver.\n\nBefore using Langfuse Cloud, set the following environment variables:\n  LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST\n\nYou can find your API keys in the Langfuse dashboard under Project Settings > API Keys.\n""")
def langfuse():
    ... # existing code 