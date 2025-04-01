"""
Crawler script for documentation pages.
This script crawls documentation pages and stores them in a vector database.
"""
import asyncio
import os
import sys

# Make sure the project root is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.crawl import crawl_and_store, get_urls_from_sitemap
from utils.db import create_supabase_client
from config.settings import get_settings

# Import OpenAI client
from openai import AsyncOpenAI, AzureOpenAI

async def main():
    """Main entry point for the crawler"""
    # Get settings
    settings = get_settings()
    
    # Validate required settings
    missing = settings.validate()
    if missing:
        print(f"Missing required settings: {', '.join(missing)}")
        return
    
    # Create clients
    supabase = create_supabase_client()
    
    # Initialize OpenAI client (may not be used if Azure is available)
    openai_client = None
    if settings.OPENAI_API_KEY:
        openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    
    # Get URLs to crawl
    print("Getting URLs from sitemap...")
    sitemap_url = "https://ai.pydantic.dev/sitemap.xml"
    urls = get_urls_from_sitemap(sitemap_url)
    
    if not urls:
        print("No URLs found in sitemap.")
        return
    
    print(f"Found {len(urls)} URLs to crawl.")
    
    # Determine which models to use
    embedding_model = None  # Use default from settings
    llm_model = settings.AZURE_OPENAI_O1_DEPLOYMENT  # Use O1 for title/summary generation
    
    # Crawl and store
    print("Starting crawl...")
    await crawl_and_store(
        urls,
        supabase,
        openai_client,
        table_name="site_pages",
        embedding_model=embedding_model,
        llm_model=llm_model,
        max_concurrent=settings.MAX_CONCURRENT_REQUESTS,
        chunk_size=settings.DEFAULT_CHUNK_SIZE
    )
    
    print("Crawl complete.")

if __name__ == "__main__":
    asyncio.run(main()) 