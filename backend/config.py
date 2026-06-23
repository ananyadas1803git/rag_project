"""
Configuration module for RAG chatbot backend.
Handles environment variables and application settings.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # LLM Configuration
    llm_provider: str = Field(default="openai", description="LLM provider: 'openai' or 'ollama'")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-3.5-turbo", description="OpenAI model name")
    ollama_model: str = Field(default="llama3", description="Ollama model name")
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama base URL")
    
    # Embedding Configuration
    embedding_provider: str = Field(default="huggingface", description="Embedding provider: 'openai' or 'huggingface'")
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Embedding model name"
    )
    
    # RAG Configuration
    chunk_size: int = Field(default=500, description="Document chunk size")
    chunk_overlap: int = Field(default=100, description="Document chunk overlap")
    top_k_retrieval: int = Field(default=3, description="Number of chunks to retrieve")
    
    # Vector Database Configuration
    vector_db_path: str = Field(
        default="backend/vectordb",
        description="Path to persist vector database"
    )
    
    # File Storage Configuration
    uploads_path: str = Field(
        default="backend/uploads",
        description="Path to store uploaded files"
    )
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    
    # CORS Configuration
    cors_origins: list[str] = Field(
        default=["http://localhost:8501", "http://localhost:8502"],
        description="Allowed CORS origins"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
