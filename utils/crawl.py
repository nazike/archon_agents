from typing import List, Dict, Any, Optional, Tuple
import asyncio
import requests
from xml.etree import ElementTree
from datetime import datetime, timezone
from urllib.parse import urlparse
import os

from supabase import Client
from openai import AsyncOpenAI, AsyncAzureOpenAI

# Import our utilities
from utils.chunking import chunk_with_metadata
from utils.embeddings import get_embedding
from utils.db import batch_insert

async def process_document(
    url: str,
    markdown: str,
    openai_client: Optional[AsyncOpenAI] = None,  # Kept for backward compatibility
    embedding_model: str = "text-embedding-3-small",
    chunk_size: int = 5000,
    title: str = ""
) -> List[Dict[str, Any]]:
    """
    Process a document by chunking it and generating embeddings
    
    Args:
        url: URL of the document
        markdown: Markdown content of the document
        openai_client: OpenAI client (not used, kept for backward compatibility)
        embedding_model: Name of the embedding model
        chunk_size: Size of each chunk
        title: Title of the document
    
    Returns:
        List of processed chunks with embeddings
    """
    # Chunk the document
    chunks = chunk_with_metadata(markdown, url, title, chunk_size)
    
    # Generate embeddings for each chunk
    for chunk in chunks:
        embedding = await get_embedding(chunk["content"], model=embedding_model)
        chunk["embedding"] = embedding
        
        # Add metadata
        if "metadata" not in chunk:
            chunk["metadata"] = {}
            
        chunk["metadata"].update({
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "url_path": urlparse(url).path,
            "embedding_model": embedding_model
        })
    
    return chunks

async def get_title_and_summary(
    chunk: str,
    url: str,
    openai_client: Optional[AsyncOpenAI] = None,
    model: str = "gpt-3.5-turbo"
) -> Dict[str, str]:
    """
    Extract title and summary using an LLM
    
    Args:
        chunk: Text chunk
        url: URL of the document
        openai_client: OpenAI client (will create Azure client if None)
        model: Model to use
    
    Returns:
        Dict with title and summary
    """
    from config.settings import get_settings
    settings = get_settings()
    
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""
    
    try:
        # Create Azure client if not provided
        if openai_client is None:
            openai_client = AsyncAzureOpenAI(
                api_key=settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
            )
        
        response = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  # Send first 1000 chars for context
            ],
            response_format={"type": "json_object"}
        )
        import json
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

def get_urls_from_sitemap(sitemap_url: str) -> List[str]:
    """
    Get URLs from a sitemap
    
    Args:
        sitemap_url: URL of the sitemap
        
    Returns:
        List of URLs
    """
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        
        # Parse the XML
        root = ElementTree.fromstring(response.content)
        
        # Extract all URLs from the sitemap
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

async def crawl_and_store(
    urls: List[str],
    supabase: Client,
    openai_client: Optional[AsyncOpenAI] = None,
    table_name: str = "site_pages",
    embedding_model: str = "text-embedding-3-small",
    llm_model: str = "gpt-3.5-turbo",
    max_concurrent: int = 5,
    chunk_size: int = 5000
):
    """
    Crawl URLs and store the results in Supabase
    
    Args:
        urls: List of URLs to crawl
        supabase: Supabase client
        openai_client: OpenAI client (optional, will use Azure if None)
        table_name: Name of the table to store the results
        embedding_model: Name of the embedding model
        llm_model: Name of the LLM model for title and summary extraction
        max_concurrent: Maximum number of concurrent requests
        chunk_size: Size of each chunk
    """
    try:
        # Import here to avoid circular imports
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
        
        browser_config = BrowserConfig(
            headless=True,
            verbose=False,
            extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
        )
        crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

        # Create the crawler instance
        crawler = AsyncWebCrawler(config=browser_config)
        await crawler.start()
        
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_url(url: str):
            async with semaphore:
                try:
                    result = await crawler.arun(
                        url=url,
                        config=crawl_config,
                        session_id="session1"
                    )
                    
                    if result.success:
                        print(f"Successfully crawled: {url}")
                        # Process the document
                        chunks = await process_document(
                            url,
                            result.markdown_v2.raw_markdown,
                            openai_client,
                            embedding_model=embedding_model,
                            chunk_size=chunk_size
                        )
                        
                        # Add title and summary to each chunk
                        for chunk in chunks:
                            extracted = await get_title_and_summary(
                                chunk["content"],
                                url,
                                openai_client,
                                model=llm_model
                            )
                            chunk["title"] = extracted["title"]
                            chunk["summary"] = extracted["summary"]
                        
                        # Store the chunks in Supabase
                        await batch_insert(supabase, table_name, chunks)
                    else:
                        print(f"Failed: {url} - Error: {result.error_message}")
                except Exception as e:
                    print(f"Error processing URL {url}: {e}")
        
        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
        
    except Exception as e:
        print(f"Error in crawl_and_store: {e}")
    finally:
        if 'crawler' in locals():
            await crawler.close() 