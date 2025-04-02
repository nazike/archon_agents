from openai import AsyncOpenAI, AsyncAzureOpenAI
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
        # Get the actual deployment name for embeddings
        deployment = model or settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        
        # Create Azure OpenAI client with proper configuration
        client = AsyncAzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
        )
        
        response = await client.embeddings.create(
            model=deployment,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting Azure embedding: {e}")
        return [0] * 1536  # Return zero vector on error

async def get_embedding(
    text: str, 
    openai_client: AsyncOpenAI = None,  # Keeping for backward compatibility
    model: str = None
) -> List[float]:
    """
    Get embedding vector from Azure OpenAI.
    
    Args:
        text: Text to get embedding for
        openai_client: Optional OpenAI client (not used, kept for compatibility)
        model: Optional embedding model to use
    
    Returns:
        Embedding vector
    """
    # We now use Azure exclusively for embeddings
    return await get_azure_embedding(text, model)

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
    openai_client: AsyncOpenAI = None,  # Keeping for backward compatibility
    query: str = "",
    match_count: int = 5,
    filter_params: Optional[Dict[str, Any]] = None,
    embedding_model: str = None
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant documents using RAG
    
    Args:
        supabase: Supabase client
        openai_client: Optional OpenAI client (not used, kept for compatibility)
        query: User query
        match_count: Number of matches to return
        filter_params: Optional filter parameters
        embedding_model: Optional embedding model to use
    
    Returns:
        List of relevant documents
    """
    # Get embedding for query
    query_embedding = await get_embedding(query, model=embedding_model)
    
    # Search for relevant documents
    return await vector_search(
        supabase,
        query_embedding,
        match_count=match_count,
        filter_params=filter_params
    ) 