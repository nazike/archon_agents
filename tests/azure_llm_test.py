import os
import asyncio
import json
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

async def test_azure_openai_connection():
    """Tests connection and basic completion with Azure OpenAI."""
    
    print("--- Azure OpenAI Connection Test ---")
    
    # Configuration - directly from environment variables
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = "2024-12-01-preview"  # Required version for o3-mini model
    deployment_name = os.getenv("AZURE_OPENAI_O3_DEPLOYMENT", "o3-mini")

    print(f"Endpoint: {azure_endpoint}")
    print(f"API Version: {api_version}")
    print(f"Deployment Name: {deployment_name}")
    
    if not all([azure_endpoint, api_key, deployment_name]):
        print("\nERROR: Missing one or more required environment variables:")
        print("- AZURE_OPENAI_ENDPOINT")
        print("- AZURE_OPENAI_API_KEY")
        print("- AZURE_OPENAI_O3_DEPLOYMENT (or ensure it's set)")
        return

    try:
        print("\nInitializing AsyncAzureOpenAI client...")
        client = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        print("Client initialized successfully.")

        print(f"\nAttempting chat completion with deployment '{deployment_name}'...")
        response = await client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Does Azure OpenAI work?"}
            ]
        )
        
        print("\nChat completion successful!")
        print("\n=== Response Details ===")
        print(f"Model: {response.model}")
        print(f"ID: {response.id}")
        print(f"Created: {response.created}")
        print("\n=== Message Content ===")
        print(f"Content: '{response.choices[0].message.content}'")
        print(f"Role: {response.choices[0].message.role}")
        print(f"Finish Reason: {response.choices[0].finish_reason}")
        print("\n=== Usage Stats ===")
        print(f"Prompt Tokens: {response.usage.prompt_tokens}")
        print(f"Completion Tokens: {response.usage.completion_tokens}")
        print(f"Total Tokens: {response.usage.total_tokens}")
        
    except Exception as e:
        print(f"\nERROR: An exception occurred:")
        print(e)
        import traceback
        traceback.print_exc()

    print("\n--- Test Complete ---")

if __name__ == "__main__":
    asyncio.run(test_azure_openai_connection()) 