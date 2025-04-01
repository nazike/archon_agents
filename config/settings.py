import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()

class Settings:
    """Centralized settings for the application"""
    
    # OpenAI settings
    OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')
    
    # Azure OpenAI settings
    AZURE_OPENAI_API_KEY: Optional[str] = os.getenv('AZURE_OPENAI_API_KEY')
    AZURE_OPENAI_ENDPOINT: Optional[str] = os.getenv('AZURE_OPENAI_ENDPOINT')
    AZURE_OPENAI_O1_DEPLOYMENT: str = os.getenv('AZURE_OPENAI_O1_DEPLOYMENT', 'o1')
    AZURE_OPENAI_O3_DEPLOYMENT: str = os.getenv('AZURE_OPENAI_O3_DEPLOYMENT', 'o3-mini')
    AZURE_OPENAI_API_VERSION: str = os.getenv('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')
    AZURE_OPENAI_EMBEDDING_ENDPOINT: Optional[str] = os.getenv('AZURE_OPENAI_EMBEDDING_ENDPOINT')
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT', 'text-embedding-3-small')
    AZURE_OPENAI_EMBEDDING_API_VERSION: str = os.getenv('AZURE_OPENAI_EMBEDDING_API_VERSION', '2023-05-15')
    
    # AWS Bedrock settings
    AWS_DEFAULT_REGION: str = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_CLAUDE_HAIKU_MODEL: str = os.getenv('AWS_CLAUDE_HAIKU_MODEL', 'anthropic.claude-3-haiku-20240307-v1:0')
    AWS_CLAUDE_SONNET_MODEL: str = os.getenv('AWS_CLAUDE_SONNET_MODEL', 'us.anthropic.claude-3-7-sonnet-20250219-v1:0')
    AWS_FALLBACK_MODEL: Optional[str] = os.getenv('AWS_FALLBACK_MODEL')
    
    # GCP settings
    VERTEXAI_PROJECT: Optional[str] = os.getenv('VERTEXAI_PROJECT')
    VERTEXAI_LOCATION: str = os.getenv('VERTEXAI_LOCATION', 'us-east5')
    GCP_GEMINI_PRO_MODEL: str = os.getenv('GCP_GEMINI_PRO_MODEL', 'gemini-1.5-pro-002')
    GCP_GEMINI_FLASH_MODEL: str = os.getenv('GCP_GEMINI_FLASH_MODEL', 'gemini-1.5-flash-002')
    GCP_CLAUDE_MODEL: str = os.getenv('GCP_CLAUDE_MODEL', 'claude-3-5-sonnet-v2@20241022')
    GEMINI_API_KEY: Optional[str] = os.getenv('GEMINI_API_KEY')
    
    # API keys
    RAPID_API_KEY: Optional[str] = os.getenv('RAPID_API_KEY')
    FINANCIAL_MODELING_PREP_API_KEY: Optional[str] = os.getenv('FINANCIAL_MODELING_PREP_API_KEY')
    
    # Postgres settings
    POSTGRES_USER: str = os.getenv('POSTGRES_USER', 'postgres')
    POSTGRES_PASSWORD: str = os.getenv('POSTGRES_PASSWORD', 'postgres')
    POSTGRES_DB: str = os.getenv('POSTGRES_DB', 'memory_db')
    POSTGRES_HOST: str = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT: int = int(os.getenv('POSTGRES_PORT', '5432'))
    
    # Supabase settings
    SUPABASE_URL: Optional[str] = os.getenv('SUPABASE_URL')
    SUPABASE_KEY: Optional[str] = os.getenv('SUPABASE_KEY')
    
    # Application settings
    WORKBENCH_DIR: str = os.getenv('WORKBENCH_DIR', 'workbench')
    DEFAULT_CHUNK_SIZE: int = int(os.getenv('DEFAULT_CHUNK_SIZE', '5000'))
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv('MAX_CONCURRENT_REQUESTS', '5'))
    
    # Model references for compatibility with the original code
    PRIMARY_MODEL: str = AZURE_OPENAI_O1_DEPLOYMENT  # Maps to o1 by default
    REASONER_MODEL: str = AZURE_OPENAI_O3_DEPLOYMENT  # Maps to o3-mini by default
    EMBEDDING_MODEL: str = AZURE_OPENAI_EMBEDDING_DEPLOYMENT  # Maps to text-embedding-3-small by default
    
    @property
    def is_ollama(self) -> bool:
        """Check if the base URL is for Ollama"""
        # This property is kept for backward compatibility
        return False
    
    def validate(self) -> list[str]:
        """
        Validate settings and return a list of missing required settings
        
        Returns:
            List of missing required settings
        """
        missing = []
        
        # Check for OpenAI or Azure credentials
        if not self.OPENAI_API_KEY and not self.AZURE_OPENAI_API_KEY:
            missing.append("Either OPENAI_API_KEY or AZURE_OPENAI_API_KEY is required")
        
        # Check for Supabase credentials
        if not self.SUPABASE_URL:
            missing.append("SUPABASE_URL is required")
        if not self.SUPABASE_KEY:
            missing.append("SUPABASE_KEY is required")
        
        return missing

# Create a singleton instance
settings = Settings()

def get_settings() -> Settings:
    """
    Get the settings instance
    
    Returns:
        Settings instance
    """
    return settings 