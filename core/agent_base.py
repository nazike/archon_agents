from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from typing import Any, Optional, Type, Literal, TypeVar
from openai import AsyncOpenAI, AsyncAzureOpenAI
import os
from dotenv import load_dotenv
import httpx

from abc import ABC, abstractmethod
from typing import Any, Generic

load_dotenv()

# Define TypeVar for client interfaces
InterfaceClient = TypeVar('InterfaceClient')

class Provider(ABC, Generic[InterfaceClient]):
    """Abstract class for a provider.

    The provider is in charge of providing an authenticated client to the API.

    Each provider only supports a specific interface. A interface can be supported by multiple providers.

    For example, the OpenAIModel interface can be supported by the OpenAIProvider and the DeepSeekProvider.
    """

    _client: InterfaceClient

    @property
    @abstractmethod
    def name(self) -> str:
        """The provider name."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def base_url(self) -> str:
        """The base URL for the provider API."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def client(self) -> InterfaceClient:
        """The client for the provider."""
        raise NotImplementedError()

# Helper function for httpx client (stub implementation)
def cached_async_http_client(provider: str) -> httpx.AsyncClient:
    """Return a cached httpx.AsyncClient for the given provider."""
    return httpx.AsyncClient()

def infer_provider(provider: str) -> Provider[Any]:
    """Infer the provider from the provider name."""
    if provider == 'openai':
        return OpenAIProvider()
    elif provider == 'azure':
        return AzureOpenAIProvider()
    elif provider == 'deepseek':
        from .deepseek import DeepSeekProvider
        return DeepSeekProvider()
    elif provider == 'google-vertex':
        from .google_vertex import GoogleVertexProvider
        return GoogleVertexProvider()
    elif provider == 'google-gla':
        from .google_gla import GoogleGLAProvider
        return GoogleGLAProvider()
    elif provider == 'bedrock':
        from .bedrock import BedrockProvider
        return BedrockProvider()
    elif provider == 'groq':
        from .groq import GroqProvider
        return GroqProvider()
    elif provider == 'anthropic':
        from .anthropic import AnthropicProvider
        return AnthropicProvider()
    elif provider == 'mistral':
        from .mistral import MistralProvider
        return MistralProvider()
    elif provider == 'cohere':
        from .cohere import CohereProvider
        return CohereProvider()
    else:
        raise ValueError(f'Unknown provider: {provider}')

class OpenAIProvider(Provider[AsyncOpenAI]):
    """Provider for OpenAI API."""

    @property
    def name(self) -> str:
        return 'openai'

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new OpenAI provider.

        Args:
            base_url: The base url for the OpenAI requests. If not provided, the `OPENAI_BASE_URL` environment variable
                will be used if available. Otherwise, defaults to OpenAI's base url.
            api_key: The API key to use for authentication, if not provided, the `OPENAI_API_KEY` environment variable
                will be used if available.
            openai_client: An existing AsyncOpenAI client to use. If provided, `base_url`, `api_key`, and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        if api_key is None and 'OPENAI_API_KEY' not in os.environ and base_url is not None and openai_client is None:
            api_key = 'api-key-not-set'

        if openai_client is not None:
            assert base_url is None, 'Cannot provide both `openai_client` and `base_url`'
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = cached_async_http_client(provider='openai')
            self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)

class AzureOpenAIProvider(Provider[AsyncAzureOpenAI]):
    """Provider for Azure OpenAI API."""

    @property
    def name(self) -> str:
        return 'azure'

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def client(self) -> AsyncAzureOpenAI:
        return self._client

    def __init__(
        self,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        azure_client: AsyncAzureOpenAI | None = None,
    ) -> None:
        """Create a new Azure OpenAI provider.

        Args:
            azure_endpoint: The Azure endpoint. If not provided, uses AZURE_OPENAI_ENDPOINT env var.
            api_version: The API version. If not provided, uses AZURE_OPENAI_API_VERSION env var.
            api_key: The API key. If not provided, uses AZURE_OPENAI_API_KEY env var.
            azure_client: An existing AsyncAzureOpenAI client to use.
        """
        if azure_client is not None:
            self._client = azure_client
        else:
            # Use environment variables with defaults matching the working test
            self._client = AsyncAzureOpenAI(
                azure_endpoint=azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=api_version or "2024-12-01-preview",  # Default to working version
                api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            )

def create_agent(
    model_name: str,
    system_prompt: str,
    provider_type: str = "openai",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    deps_type: Optional[Type] = None,
    retries: int = 3,
) -> Agent:
    """
    Create a Pydantic AI agent with the specified configuration.
    
    Args:
        model_name: Name of the model to use (or deployment name for Azure)
        system_prompt: System prompt for the agent
        provider_type: Type of provider ("openai" or "azure")
        api_key: Optional API key for the model
        base_url: Optional base URL for the model
        deps_type: Optional type for dependencies
        retries: Number of retries for failed requests
        
    Returns:
        Agent: The configured Pydantic AI agent
    """
    from core.models import get_openai_model, get_azure_openai_model
    
    # Initialize the model based on provider type
    if provider_type == "azure":
        model = get_azure_openai_model(
            deployment_name=model_name,
            api_key=api_key
        )
    else:
        model = get_openai_model(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url
        )
    
    return Agent(
        model,
        system_prompt=system_prompt,
        deps_type=deps_type,
        retries=retries
    ) 