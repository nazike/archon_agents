"""
Main entry point for the Archon Agent Builder
"""
import streamlit as st
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import settings
from config.settings import get_settings, Settings

# Import the Streamlit UI utilities
from ui.streamlit_app import create_agent_ui

# Import the Archon agent
from agents.archon.agent import archon_graph

# Get settings
settings = get_settings()

# Check required settings
missing_settings = settings.validate()
if missing_settings:
    st.error(f"Missing required settings: {', '.join(missing_settings)}")
    st.info("Please set these values in your .env file and restart the application.")
    st.stop()

# Define example prompts
example_prompts = [
    "Build me an AI agent that can search the web with the Brave API",
    "I need an agent that can summarize YouTube videos using their transcripts",
    "Create an AI assistant that can query data from a Postgres database",
    "I want to build a chatbot that can use Google Calendar API to manage appointments"
]

# Create the UI
create_agent_ui(
    title="Archon - Agent Builder",
    agent_graph=archon_graph,
    description="Describe to me an AI agent you want to build and I'll code it for you with Pydantic AI.",
    example_prompts=example_prompts
)

if __name__ == "__main__":
    # Nothing additional to do here since Streamlit handles the main loop
    pass 