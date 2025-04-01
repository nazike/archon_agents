# Archon V2 - Agentic Workflow for Building Pydantic AI Agents

This is the second iteration of the Archon project, building upon V1 by introducing LangGraph for a full agentic workflow. The system starts with a reasoning LLM (like O3-mini or R1) that analyzes user requirements and documentation to create a detailed scope, which then guides specialized coding and routing agents in generating high-quality Pydantic AI agents.

An intelligent documentation crawler and RAG (Retrieval-Augmented Generation) system built using Pydantic AI, LangGraph, and Supabase that is capable of building other Pydantic AI agents. The system crawls the Pydantic AI documentation, stores content in a vector database, and provides Pydantic AI agent code by retrieving and analyzing relevant documentation chunks.

This version also supports local LLMs with Ollama for the main agent and reasoning LLM.

Note that we are still relying on OpenAI for embeddings no matter what, but future versions of Archon will change that.

## Project Structure

The project is organized in a modular way to enable easy creation of multiple agents:

```
project/
├── agents/               # All agent implementations
│   ├── archon/           # Original Archon agent
│   │   ├── __init__.py
│   │   ├── agent.py      # Archon agent implementation
│   │   └── prompts.py    # System prompts for Archon
│   └── your_new_agent/   # Add your own agents here
├── core/                 # Core functionality
│   ├── agent_base.py     # Agent creation utilities
│   ├── graph.py          # LangGraph workflow utilities
│   ├── models.py         # LLM model utilities
│   └── state.py          # Agent state management
├── utils/                # Utility functions
│   ├── chunking.py       # Text chunking utilities
│   ├── crawl.py          # Document crawling utilities
│   ├── db.py             # Database operations
│   └── embeddings.py     # Vector embedding utilities
├── ui/                   # User interfaces
│   └── streamlit_app.py  # Streamlit UI components
├── config/               # Configuration
│   └── settings.py       # Settings manager
├── scripts/              # Utility scripts
│   ├── crawler.py        # Documentation crawler script
│   └── setup_db.py       # Database setup script
├── db/                   # Database scripts
│   ├── site_pages.sql    # SQL for setting up the database
│   └── ollama_site_pages.sql  # Ollama-specific SQL
├── main.py               # Main entry point
├── requirements.txt      # Project dependencies
└── .env.example          # Example environment variables
```

## Features

- Multi-agent workflow using LangGraph
- Specialized agents for reasoning, routing, and coding
- Pydantic AI documentation crawling and chunking
- Vector database storage with Supabase
- Semantic search using OpenAI embeddings
- RAG-based question answering
- Support for code block preservation
- Streamlit UI for interactive querying

## Prerequisites

- Python 3.11+
- Supabase account and database
- OpenAI/OpenRouter API key or Ollama for local LLMs
- Streamlit (for web interface)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/coleam00/archon.git
cd archon
```

2. Install dependencies (recommended to use a Python virtual environment):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up environment variables:
   - Rename `.env.example` to `.env`
   - Edit `.env` with your API keys and preferences:
   ```env
   BASE_URL=https://api.openai.com/v1 for OpenAI, https://api.openrouter.ai/v1 for OpenRouter, or your Ollama URL
   LLM_API_KEY=your_openai_or_openrouter_api_key
   OPENAI_API_KEY=your_openai_api_key
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_KEY=your_supabase_service_key
   PRIMARY_MODEL=gpt-4o-mini  # or your preferred OpenAI model for main agent
   REASONER_MODEL=o3-mini     # or your preferred OpenAI model for reasoning
   ```

## Usage

### Database Setup

Use the setup script to create the necessary tables:

```bash
python scripts/setup_db.py
```

For Ollama-specific setup:

```bash
python scripts/setup_db.py --ollama
```

### Crawl Documentation

To crawl and store documentation in the vector database:

```bash
python scripts/crawler.py
```

This will:
1. Fetch URLs from the documentation sitemap
2. Crawl each page and split into chunks
3. Generate embeddings and store in Supabase

### Running the Agent

Launch the Streamlit web interface:

```bash
streamlit run main.py
```

The interface will be available at `http://localhost:8501`

## Creating Your Own Agent

To create a new agent:

1. Create a new directory in the `agents` folder:
```bash
mkdir -p agents/your_agent_name
```

2. Create the necessary files:
```
agents/your_agent_name/
├── __init__.py
├── agent.py      # Your agent implementation
└── prompts.py    # System prompts for your agent
```

3. Implement your agent using the core components
4. Create a custom main file or add your agent to the existing one

See the `agents/archon` directory for an example implementation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
