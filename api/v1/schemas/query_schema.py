"""
Pydantic schemas for RAG query/request/response.
Compatible with ingestion.py metadata structure.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal


class MetadataEntry(BaseModel):
    """
    Metadata for a single retrieved chunk.
    Matches the structure produced by ingestion.py.
    """
    # Ranking & scoring
    rank: int = Field(..., ge=1, description="1-based rank in retrieval results")
    cosine_similarity: Optional[float] = Field(None, ge=0, le=1, description="TF-IDF cosine similarity")
    bm25: Optional[float] = Field(None, ge=0, le=1, description="BM25 score (normalized)")
    relevance_score: Optional[float] = Field(None, ge=0, le=1, description="Composite relevance score")
    rerank_score: Optional[float] = Field(None, ge=0, le=1, description="Cohere rerank score")
    rerank_index: Optional[int] = Field(None, ge=0, description="Original position before reranking")
    
    # Content & citation
    citation: str = Field(..., description="Verbatim excerpt or key sentence from chunk")
    content: str = Field(..., description="Full searchable text of the chunk")
    
    # Ingestion metadata (from ingestion.py)
    document_id: Optional[str] = Field(None, description="Unique document UUID")
    document_name: Optional[str] = Field(None, description="Source filename")
    page_number: Optional[int] = Field(None, ge=1, description="Page number in original PDF")
    chunk_index: Optional[int] = Field(None, ge=0, description="Chunk index within document")
    
    # Modality & structure
    modality: Literal["text", "table", "image"] = Field(default="text", description="Chunk type")
    element_type: Optional[str] = Field(None, description="Docling element label")
    section_header: Optional[str] = Field(None, description="Document section this chunk belongs to")
    docling_label: Optional[str] = Field(None, description="Docling's label for this element")
    
    # Multimodal-specific fields
    docling_caption: Optional[str] = Field(None, description="Caption extracted by Docling")
    llm_generated_caption: Optional[str] = Field(None, description="AI-generated description")
    table_data: Optional[str] = Field(None, description="Table content in text form")
    table_title: Optional[str] = Field(None, description="Table title/caption")
    
    # Spatial & temporal
    bbox: Optional[List[float]] = Field(None, description="[left, top, right, bottom] coordinates")
    source_file: Optional[str] = Field(None, description="Original file path")
    ingested_at: Optional[str] = Field(None, description="ISO timestamp of ingestion")
    
    # Allow additional fields from DB
    class Config:
        extra = "allow"


class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    query: str = Field(..., min_length=1, max_length=2000, description="User's natural language question")
    k: int = Field(default=5, ge=1, le=100, description="Number of top-k results to retrieve")


class QueryResponse(BaseModel):
    """
    Response model for RAG queries.
    
    The metadata list contains enriched information from:
    1. ingestion.py (document structure, multimodal content)
    2. scoring.py (cosine_sim, BM25, relevance_score)
    3. reranker.py (rerank_score, rerank_index)
    """
    query: str = Field(..., description="The original user query")
    answer: str = Field(..., description="Synthesized answer from retrieved chunks")
    metadata: List[MetadataEntry] = Field(
        default_factory=list,
        description="List of metadata entries for each retrieved chunk"
    )