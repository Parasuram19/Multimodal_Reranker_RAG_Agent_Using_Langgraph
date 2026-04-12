"""
Core helper functions for vector store and embeddings.
"""

import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv(override=True)

PG_CONNECTION = os.getenv("PG_CONNECTION_STRING") or os.getenv("DATABASE_URL")
if not PG_CONNECTION:
    raise ValueError("PG_CONNECTION_STRING or DATABASE_URL is not set")

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set")

EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDINGS_MODEL", "gemini-embedding-2-preview")


def get_embedding_model():
    """Get the Gemini embeddings model instance."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
        output_dimensionality=1536
    )


def get_vector_store(collection_name: str = "financial_rag"):
    """
    Get the PGVector store instance.
    
    Args:
        collection_name: Name of the PGVector collection (default: "policy_docs")
    
    Returns:
        PGVector instance configured with Gemini embeddings
    """
    return PGVector(
        collection_name=collection_name,
        connection=PG_CONNECTION,
        embeddings=get_embedding_model(),
        use_jsonb=True  # Store metadata as JSONB for flexible querying
    )