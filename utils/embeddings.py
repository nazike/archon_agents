from openai import AsyncOpenAI, AzureOpenAI
from typing import List, Dict, Any, Optional
from supabase import Client
import os
from config.settings import get_settings

# Get settings
settings = get_settings()

async def get_azure_embedding(
    text: str,
    model: str = None
) -> List[float]:
    """
    Get embedding vector from Azure OpenAI.
    
    Args:
        text: Text to get embedding for
        model: Optional model override
    
    Returns:
        Embedding vector
    """
    try:
        # Create Azure OpenAI client
        client = AzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_EMBEDDING_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_EMBEDDING_ENDPOINT
        )
        
        response = client.embeddings.create(
            model=model or settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting Azure embedding: {e}")
        return [0] * 1536  # Return zero vector on error

async def get_embedding(
    text: str, 
    openai_client: AsyncOpenAI = None,
    model: str = None
) -> List[float]:
    """
    Get embedding vector from OpenAI or Azure OpenAI.
    
    Args:
        text: Text to get embedding for
        openai_client: Optional OpenAI client
        model: Optional embedding model to use
    
    Returns:
        Embedding vector
    """
    # Try to use Azure OpenAI first if credentials are available
    if settings.AZURE_OPENAI_API_KEY and settings.AZURE_OPENAI_EMBEDDING_ENDPOINT:
        return await get_azure_embedding(text, model)
    
    # Fall back to standard OpenAI
    try:
        # If no client is provided, create one
        if openai_client is None:
            openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
        response = await openai_client.embeddings.create(
            model=model or "text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

async def vector_search(
    supabase: Client,
    query_embedding: List[float],
    match_count: int = 5,
    table_name: str = "site_pages",
    function_name: str = "match_site_pages",
    filter_params: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Perform vector similarity search in Supabase
    
    Args:
        supabase: Supabase client
        query_embedding: Query embedding vector
        match_count: Number of matches to return
        table_name: Name of the table to search
        function_name: Name of the RPC function to call
        filter_params: Optional filter parameters
    
    Returns:
        List of matching documents
    """
    try:
        params = {
            "query_embedding": query_embedding,
            "match_count": match_count
        }
        
        if filter_params:
            params["filter"] = filter_params
            
        result = supabase.rpc(function_name, params).execute()
        
        if not result.data:
            return []
            
        return result.data
    except Exception as e:
        print(f"Error in vector search: {e}")
        return []

async def retrieve_relevant_documents(
    supabase: Client,
    openai_client: AsyncOpenAI = None,
    query: str = "",
    match_count: int = 5,
    filter_params: Optional[Dict[str, Any]] = None,
    embedding_model: str = None
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant documents using RAG
    
    Args:
        supabase: Supabase client
        openai_client: Optional OpenAI client (can be None if Azure is used)
        query: User query
        match_count: Number of matches to return
        filter_params: Optional filter parameters
        embedding_model: Optional embedding model to use
    
    Returns:
        List of relevant documents
    """
    # Get embedding for query
    query_embedding = await get_embedding(query, openai_client, model=embedding_model)
    
    # Search for relevant documents
    return await vector_search(
        supabase,
        query_embedding,
        match_count=match_count,
        filter_params=filter_params
    ) 