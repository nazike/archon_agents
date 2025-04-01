from pydantic_ai.models.openai import OpenAIModel
from typing import Optional, Dict, Any
import os

def get_openai_model(
    model_name: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Any]] = None
) -> OpenAIModel:
    """
    Get an OpenAI model with the given configuration
    
    Args:
        model_name: Name of the model to use
        api_key: Optional API key (defaults to OPENAI_API_KEY env var)
        base_url: Optional base URL (defaults to OpenAI's API URL)
        model_kwargs: Optional additional kwargs for the model
    
    Returns:
        An OpenAIModel instance
    """
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not base_url:
        base_url = os.getenv("BASE_URL", "https://api.openai.com/v1")
    
    kwargs = model_kwargs or {}
    
    return OpenAIModel(
        model_name,
        base_url=base_url,
        api_key=api_key,
        **kwargs
    )

def is_ollama_url(url: str) -> bool:
    """
    Check if the given URL is for Ollama
    
    Args:
        url: URL to check
    
    Returns:
        True if the URL is for Ollama, False otherwise
    """
    return "localhost" in url.lower() or "ollama" in url.lower() 