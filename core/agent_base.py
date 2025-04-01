from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from typing import Any, Optional, Type

def create_agent(
    model_name: str,
    system_prompt: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    deps_type: Optional[Type] = None,
    retries: int = 2
):
    """
    Factory function to create agents with common configuration
    
    Args:
        model_name: Name of the LLM model
        system_prompt: System prompt for the agent
        base_url: Optional base URL for API endpoint
        api_key: Optional API key
        deps_type: Optional dependency type for the agent
        retries: Number of retries for model calls
    
    Returns:
        A configured Pydantic AI agent
    """
    model = OpenAIModel(model_name, base_url=base_url, api_key=api_key)
    return Agent(
        model,
        system_prompt=system_prompt,
        deps_type=deps_type,
        retries=retries
    ) 