from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI, AsyncAzureOpenAI
# Attempt to import OpenAIProvider from agent_base
# If this causes circular imports, we might need restructuring
try:
    from core.agent_base import OpenAIProvider 
except ImportError:
    # Define a placeholder if import fails, though this shouldn't happen with current structure
    print("WARN: Could not import OpenAIProvider from core.agent_base. Ensure it's defined there.")
    class OpenAIProvider:
        def __init__(self, openai_client=None, **kwargs): pass

from typing import Optional, Dict, Any
import os

def get_openai_model(
    model_name: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Any]] = None
) -> OpenAIModel:
    """
    Get a standard OpenAI model instance.
    Use get_azure_openai_model for Azure.

    Args:
        model_name: Name of the model to use.
        api_key: Optional API key (defaults to OPENAI_API_KEY env var).
        base_url: Optional base URL (defaults to OpenAI's API URL).
        model_kwargs: Optional additional kwargs for the model.

    Returns:
        An OpenAIModel instance.
    """
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not base_url:
        base_url = os.getenv("BASE_URL", "https://api.openai.com/v1")

    kwargs = model_kwargs or {}

    return OpenAIModel(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        **kwargs
    )

def get_azure_openai_model(
    deployment_name: str,
    api_key: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    api_version: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Any]] = None
) -> OpenAIModel:
    """
    Get an OpenAIModel instance configured explicitly for Azure OpenAI.

    Args:
        deployment_name: Name of the Azure deployment (e.g., "o3-mini").
        api_key: Optional API key (defaults to AZURE_OPENAI_API_KEY env var).
        azure_endpoint: Optional Azure endpoint (defaults to AZURE_OPENAI_ENDPOINT env var).
        api_version: Optional API version (defaults to AZURE_OPENAI_API_VERSION env var).
        model_kwargs: Optional additional kwargs for the OpenAIModel itself.

    Returns:
        An OpenAIModel instance configured for Azure.
    """
    # Get config from environment if not provided
    if not api_key:
        api_key = os.getenv("AZURE_OPENAI_API_KEY")

    if not azure_endpoint:
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    if not api_version:
        api_version = "2024-12-01-preview"  # Default to the working version

    if not all([api_key, azure_endpoint, api_version]):
        raise ValueError(
            "Missing Azure configuration. Ensure AZURE_OPENAI_API_KEY, "
            "AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_API_VERSION are set in environment."
        )

    # Create the Azure OpenAI client
    azure_client = AsyncAzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
    )

    kwargs = model_kwargs or {}
    
    # Initialize OpenAIModel with Azure configuration
    model = OpenAIModel(
        model_name=deployment_name,
        api_key=api_key,
        base_url=azure_endpoint,
        **kwargs
    )
    
    # Set the client directly on the model instance
    model.client = azure_client
    return model

def is_ollama_url(url: str) -> bool:
    """
    Check if the given URL is for Ollama
    
    Args:
        url: URL to check
    
    Returns:
        True if the URL is for Ollama, False otherwise
    """
    return "localhost" in url.lower() or "ollama" in url.lower() 