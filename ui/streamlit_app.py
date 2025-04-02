import streamlit as st
import asyncio
import uuid
import os
from typing import Any, Callable, AsyncGenerator

# Import our configuration
from config.settings import get_settings

# For type hints only
from langgraph.graph import StateGraph
from typing import TypeVar, Generic, Dict, Any

# Type for graph
T = TypeVar('T')
class CompilableGraph(Generic[T]):
    """Type hint for a compiled graph that can be used with astream"""
    async def astream(self, *args, **kwargs) -> AsyncGenerator[str, None]:
        ...

# Set page configuration
st.set_page_config(
    page_title="Agent Builder",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

settings = get_settings()

@st.cache_resource
def get_thread_id() -> str:
    """Generate a unique thread ID for the conversation"""
    return str(uuid.uuid4())

# Keep a unique thread ID per session
thread_id = get_thread_id()

async def stream_agent_response(
    agent_graph: CompilableGraph,
    user_input: str,
    first_message: bool = False
) -> AsyncGenerator[str, None]:
    """
    Stream the agent's response for a user input
    
    Args:
        agent_graph: Compiled LangGraph to run
        user_input: User input message
        first_message: Whether this is the first message
    
    Yields:
        Chunks of the agent's response
    """
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    if first_message:
        # First message from user
        async for msg in agent_graph.astream(
            {"latest_user_message": user_input}, 
            config, 
            stream_mode="custom"
        ):
            yield msg
    else:
        # Continue the conversation
        from langgraph.types import Command
        async for msg in agent_graph.astream(
            Command(resume=user_input), 
            config, 
            stream_mode="custom"
        ):
            yield msg

def create_agent_ui(
    title: str,
    agent_graph: CompilableGraph,
    description: str = "",
    example_prompts: list[str] = None
):
    """
    Create a Streamlit UI for an agent
    
    Args:
        title: Title of the agent
        agent_graph: Compiled LangGraph for the agent
        description: Description of the agent
        example_prompts: Example prompts for the agent
    """
    st.title(title)
    
    if description:
        st.write(description)
    
    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        message_type = message["type"]
        if message_type in ["human", "ai", "system"]:
            with st.chat_message(message_type):
                st.markdown(message["content"])

    # Show example prompts
    if example_prompts:
        with st.expander("Example prompts"):
            for prompt in example_prompts:
                if st.button(prompt):
                    # Use the example as user input
                    add_user_message(prompt)
                    run_agent(agent_graph, prompt)

    # Chat input for the user
    user_input = st.chat_input("Enter your message...")

    if user_input:
        add_user_message(user_input)
        run_agent(agent_graph, user_input)

def add_user_message(message: str):
    """Add a user message to the chat history"""
    st.session_state.messages.append({"type": "human", "content": message})
    
    # Display the message
    with st.chat_message("user"):
        st.markdown(message)

def run_agent(agent_graph: CompilableGraph, user_input: str):
    """Run the agent with the user input"""
    first_message = len(st.session_state.messages) <= 1
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Create and run the async loop
        async def process_response():
            nonlocal full_response
            async for chunk in stream_agent_response(agent_graph, user_input, first_message):
                full_response += chunk
                # Update the placeholder with the current response content
                message_placeholder.markdown(full_response)
        
        # Run in asyncio
        asyncio.run(process_response())
        
        # Add the full response to the chat history
        st.session_state.messages.append({"type": "ai", "content": full_response})

# This allows the file to be imported or run directly
if __name__ == "__main__":
    st.warning("This is a UI module that should be imported by an agent. Run main.py instead.")
    st.info("To create a new agent UI, import this module and use the create_agent_ui function.")
    
    # Display a placeholder UI
    st.markdown("## Agent Builder Platform")
    st.markdown("""
    This is the core UI module for the Agent Builder Platform.
    
    To create your own agent:
    1. Create a new agent module in the `agents` directory
    2. Import this module and use `create_agent_ui`
    3. Run your agent's main file
    """) 