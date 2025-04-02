"""
Archon Agent - An agentic workflow for building Pydantic AI agents
"""
from pydantic_ai import Agent, RunContext
from typing import TypedDict, List, Annotated, Dict, Any, Optional
from dataclasses import dataclass
from openai import AsyncOpenAI, AsyncAzureOpenAI
from supabase import Client
import os
import asyncio

# Import core components
from core.agent_base import create_agent
from core.graph import create_agent_graph, create_interrupt_node
from core.state import ReasoningAgentState

# Import agent prompts
from agents.archon.prompts import (
    REASONER_SYSTEM_PROMPT,
    CODER_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    END_CONVERSATION_SYSTEM_PROMPT
)

# Import from settings
from config.settings import get_settings

# Import utilities
from utils.embeddings import get_embedding, retrieve_relevant_documents
from utils.db import create_supabase_client

# Import Pydantic AI message utilities
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter
)

# Import from models
from core.models import get_azure_openai_model

# Get settings
settings = get_settings()

@dataclass
class PydanticAIDeps:
    """Dependencies for the Pydantic AI coder agent"""
    supabase: Client
    reasoner_output: str
    openai_client: Optional[AsyncOpenAI] = None

class ArchonState(ReasoningAgentState):
    """State for the Archon agent workflow"""
    relevant_docs: List[Dict[str, Any]]
    doc_sources: List[str]

# Create the agents
def create_reasoner_agent():
    """Create the reasoning agent"""
    # Use the Azure-specific model creation function
    # Pass the specific deployment name 'o3-mini'
    model = get_azure_openai_model(
        deployment_name="o3-mini", # Specific deployment name
        api_key=settings.AZURE_OPENAI_API_KEY # Explicitly pass API key for clarity/override
    )
    agent = Agent(model, system_prompt=REASONER_SYSTEM_PROMPT, retries=2)
    return agent

def create_router_agent():
    """Create the router agent"""
    # Use the Azure-specific model creation function
    # Pass the specific deployment name 'o3-mini'
    model = get_azure_openai_model(
        deployment_name="o3-mini", # Specific deployment name
        api_key=settings.AZURE_OPENAI_API_KEY
    )
    agent = Agent(model, system_prompt=ROUTER_SYSTEM_PROMPT, retries=2)
    return agent

def create_coder_agent():
    """Create the coder agent"""
    # Use the Azure-specific model creation function
    # Pass the specific deployment name 'o3-mini'
    model = get_azure_openai_model(
        deployment_name="o3-mini", # Specific deployment name
        api_key=settings.AZURE_OPENAI_API_KEY
    )
    agent = Agent(model, system_prompt=CODER_SYSTEM_PROMPT, deps_type=PydanticAIDeps, retries=2)
    
    @agent.system_prompt  
    def add_reasoner_output(ctx: RunContext[PydanticAIDeps]) -> str:
        """Add the reasoner output to the system prompt"""
        return f"""
        \n\nAdditional thoughts/instructions from the reasoner LLM. 
        This scope includes documentation pages for you to search as well: 
        {ctx.deps.reasoner_output}
        """
    
    @agent.tool
    async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
        """
        Retrieve relevant documentation chunks based on the query with RAG.
        
        Args:
            ctx: The context including the Supabase client and OpenAI client
            user_query: The user's question or query
            
        Returns:
            A formatted string containing the top 5 most relevant documentation chunks
        """
        try:
            # Get the embedding for the query
            query_embedding = await get_embedding(user_query)
            
            # Query Supabase for relevant documents
            result = ctx.deps.supabase.rpc(
                'match_site_pages',
                {
                    'query_embedding': query_embedding,
                    'match_count': 5,
                    'filter': {'source': 'pydantic_ai_docs'}
                }
            ).execute()
            
            if not result.data:
                return "No relevant documentation found."
                
            # Format the results
            formatted_chunks = []
            for doc in result.data:
                chunk_text = f"""
    # {doc['title']}
    
    {doc['content']}
    """
                formatted_chunks.append(chunk_text)
                
            # Join all chunks with a separator
            return "\n\n---\n\n".join(formatted_chunks)
            
        except Exception as e:
            print(f"Error retrieving documentation: {e}")
            return f"Error retrieving documentation: {str(e)}"
    
    @agent.tool
    async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
        """
        Retrieve a list of all available Pydantic AI documentation pages.
        
        Returns:
            List[str]: List of unique URLs for all documentation pages
        """
        try:
            # Query Supabase for unique URLs where source is pydantic_ai_docs
            result = ctx.deps.supabase.from_('site_pages') \
                .select('url') \
                .eq('metadata->>source', 'pydantic_ai_docs') \
                .execute()
            
            if not result.data:
                return []
                
            # Extract unique URLs
            urls = sorted(set(doc['url'] for doc in result.data))
            return urls
            
        except Exception as e:
            print(f"Error retrieving documentation pages: {e}")
            return []
    
    @agent.tool
    async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
        """
        Retrieve the full content of a specific documentation page by combining all its chunks.
        
        Args:
            ctx: The context including the Supabase client
            url: The URL of the page to retrieve
            
        Returns:
            str: The complete page content with all chunks combined in order
        """
        try:
            # Query Supabase for all chunks of this URL, ordered by chunk_number
            result = ctx.deps.supabase.from_('site_pages') \
                .select('title, content, chunk_number') \
                .eq('url', url) \
                .eq('metadata->>source', 'pydantic_ai_docs') \
                .order('chunk_number') \
                .execute()
            
            if not result.data:
                return f"No content found for URL: {url}"
                
            # Get title from first chunk
            title = result.data[0]['title']
            
            # Combine chunks in order
            chunks = [f"# {title}\n\n{doc['content']}" for doc in result.data]
            combined = "\n\n".join(chunks)
            
            return combined
            
        except Exception as e:
            print(f"Error retrieving page content: {e}")
            return f"Error retrieving page content: {str(e)}"
    
    return agent

def create_end_conversation_agent():
    """Create the end conversation agent"""
    # Use the Azure-specific model creation function
    # Pass the specific deployment name 'o3-mini'
    model = get_azure_openai_model(
        deployment_name="o3-mini", # Specific deployment name
        api_key=settings.AZURE_OPENAI_API_KEY
    )
    agent = Agent(model, system_prompt=END_CONVERSATION_SYSTEM_PROMPT, retries=2)
    return agent

# Node implementations
async def define_scope_with_reasoner(state: ArchonState):
    """
    Define the scope of the agent using the reasoner LLM
    
    Args:
        state: Current agent state
    
    Returns:
        Updated state
    """
    # Initialize clients
    supabase = create_supabase_client()
    
    # Get documentation pages
    try:
        result = supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'pydantic_ai_docs') \
            .execute()
        
        if result.data:
            doc_pages = sorted(set(doc['url'] for doc in result.data))
            documentation_pages_str = "\n".join(doc_pages)
        else:
            documentation_pages_str = "No documentation pages found."
    except Exception as e:
        print(f"Error getting documentation pages: {e}")
        documentation_pages_str = f"Error getting documentation pages: {str(e)}"

    # Create the reasoner agent
    reasoner = create_reasoner_agent()
    
    # Build the prompt
    prompt = f"""
    User AI Agent Request: {state['latest_user_message']}
    
    Create detailed scope document for the AI agent including:
    - Architecture diagram
    - Core components
    - External dependencies
    - Testing strategy

    Also based on these documentation pages available:

    {documentation_pages_str}

    Include a list of documentation pages that are relevant to creating this agent for the user in the scope document.
    """

    # Run the reasoner
    result = await reasoner.run(prompt)
    scope = result.data

    # Create the workbench directory if needed
    os.makedirs(settings.WORKBENCH_DIR, exist_ok=True)
    
    # Save the scope to a file
    scope_path = os.path.join(settings.WORKBENCH_DIR, "scope.md")
    with open(scope_path, "w", encoding="utf-8") as f:
        f.write(scope)
    
    # Return updated state
    return {
        "scope": scope,
        "doc_sources": doc_pages if 'doc_pages' in locals() else []
    }

async def coder_agent_node(state: ArchonState, writer):
    """
    Run the coder agent
    
    Args:
        state: Current agent state
        writer: Stream writer
    
    Returns:
        Updated state
    """
    # Initialize clients
    supabase = create_supabase_client()
    
    # Use None for openai_client, as we're directly configuring Azure in get_embedding
    openai_client = None
    
    # Prepare dependencies
    deps = PydanticAIDeps(
        supabase=supabase,
        openai_client=openai_client,
        reasoner_output=state['scope']
    )

    # Get the message history into the format for Pydantic AI
    message_history: list[ModelMessage] = []
    for message_row in state['messages']:
        message_history.extend(ModelMessagesTypeAdapter.validate_json(message_row))

    # Create the coder agent
    agent = create_coder_agent()
    
    # Run the agent and stream the output
    if settings.is_ollama:
        # For Ollama, we can't stream effectively
        result = await agent.run(
            state['latest_user_message'], 
            deps=deps, 
            message_history=message_history
        )
        writer(result.data)
    else:
        # For OpenAI, we can stream
        async with agent.run_stream(
            state['latest_user_message'],
            deps=deps,
            message_history=message_history
        ) as result:
            # Stream partial text as it arrives
            async for chunk in result.stream_text(delta=True):
                writer(chunk)

    # Return updated state
    return {"messages": [result.new_messages_json()]}

async def router_node(state: ArchonState):
    """
    Route the user message to the appropriate next node
    
    Args:
        state: Current agent state
    
    Returns:
        Next node to route to
    """
    # Create the router agent
    router = create_router_agent()
    
    # Build the prompt
    prompt = f"""
    The user has sent a message: 
    
    {state['latest_user_message']}

    If the user wants to end the conversation, respond with just the text "finish_conversation".
    If the user wants to continue coding the AI agent, respond with just the text "coder_agent".
    """

    # Run the router
    result = await router.run(prompt)
    next_action = result.data.strip()

    # Return the routing decision
    if next_action == "finish_conversation":
        return "finish_conversation"
    else:
        return "coder_agent"

async def end_conversation_node(state: ArchonState, writer):
    """
    End the conversation
    
    Args:
        state: Current agent state
        writer: Stream writer
    
    Returns:
        Updated state
    """
    # Create the end conversation agent
    agent = create_end_conversation_agent()
    
    # Get the message history into the format for Pydantic AI
    message_history: list[ModelMessage] = []
    for message_row in state['messages']:
        message_history.extend(ModelMessagesTypeAdapter.validate_json(message_row))

    # Run the agent and stream the output
    if settings.is_ollama:
        # For Ollama, we can't stream effectively
        result = await agent.run(
            state['latest_user_message'], 
            message_history=message_history
        )
        writer(result.data)
    else:
        # For OpenAI, we can stream
        async with agent.run_stream(
            state['latest_user_message'],
            message_history=message_history
        ) as result:
            # Stream partial text as it arrives
            async for chunk in result.stream_text(delta=True):
                writer(chunk)

    # Return updated state
    return {"messages": [result.new_messages_json()]}

# Create the graph
def create_archon_graph():
    """
    Create the Archon agent graph
    
    Returns:
        Compiled LangGraph
    """
    # Create the interrupt node
    get_next_user_message = create_interrupt_node("latest_user_message")
    
    # Define nodes
    nodes = {
        "define_scope_with_reasoner": define_scope_with_reasoner,
        "coder_agent": coder_agent_node,
        "get_next_user_message": get_next_user_message,
        "finish_conversation": end_conversation_node
    }
    
    # Define edges
    edges = [
        ("START", "define_scope_with_reasoner"),
        ("define_scope_with_reasoner", "coder_agent"),
        ("coder_agent", "get_next_user_message"),
        ("finish_conversation", "END")
    ]
    
    # Define conditional edges
    conditional_edges = {
        "get_next_user_message": (router_node, {
            "coder_agent": "coder_agent",
            "finish_conversation": "finish_conversation"
        })
    }
    
    # Create and return the graph
    return create_agent_graph(
        ArchonState,
        nodes,
        edges,
        conditional_edges
    )

# Create the agent graph
archon_graph = create_archon_graph() 