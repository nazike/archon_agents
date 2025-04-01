from typing import TypedDict, List, Annotated, Optional, Dict, Any
from typing_extensions import NotRequired

class BaseAgentState(TypedDict):
    """Base state for all agents"""
    latest_user_message: str
    messages: Annotated[List[bytes], lambda x, y: x + y]

class DocumentationAgentState(BaseAgentState):
    """State for agents that use documentation"""
    doc_sources: NotRequired[List[str]]
    query_embedding: NotRequired[List[float]]
    relevant_chunks: NotRequired[List[Dict[str, Any]]]

class ReasoningAgentState(BaseAgentState):
    """State for agents that use reasoning"""
    scope: str
    reasoning_steps: NotRequired[List[str]]

def create_initial_state(user_message: str) -> BaseAgentState:
    """
    Create an initial state for an agent
    
    Args:
        user_message: Initial user message
    
    Returns:
        Initial agent state
    """
    return {
        "latest_user_message": user_message,
        "messages": []
    }

def create_documentation_state(user_message: str) -> DocumentationAgentState:
    """
    Create an initial state for a documentation agent
    
    Args:
        user_message: Initial user message
    
    Returns:
        Initial documentation agent state
    """
    base_state = create_initial_state(user_message)
    return {
        **base_state,
        "doc_sources": [],
        "relevant_chunks": []
    }

def create_reasoning_state(user_message: str) -> ReasoningAgentState:
    """
    Create an initial state for a reasoning agent
    
    Args:
        user_message: Initial user message
    
    Returns:
        Initial reasoning agent state
    """
    base_state = create_initial_state(user_message)
    return {
        **base_state,
        "scope": "",
        "reasoning_steps": []
    } 