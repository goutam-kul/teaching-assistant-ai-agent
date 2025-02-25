import os
from pathlib import Path
from functools import lru_cache
from typing import List, Dict
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv(override=True)

class Settings(BaseSettings):
    """Application settings with validations"""

    # Ollama 
    OLLAMA_HOST: str = "http://localhost:11434"

    # LLM Configuration
    LLM_MODEL: str = "mistral:7b"
    LLM_TEMPERATURE: float = 0.1

    # Document processing configs
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_CONTEXT_CHUNKS: int = 3

    # ChromaDB settings
    CHROMA_SETTINGS: dict = {
            "chroma_db_impl": "duckdb+parquet",
            "anonymized_telemetry": False
        }
    
    # Embedding model configs
    EMBEDDING_MODEL: str = "BAAI/bge-small-en"

    # LLama Parse
    LLAMA_CLOUD_API_KEY: str = os.getenv("LLAMA_CLOUD_API_KEY")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True
    )

# Caching for this call 
@lru_cache()
def get_settings() -> Settings :
    """Get cached settings
    Returns:
        Settings: Application status
    """
    return Settings()