"""
Cohere Rerank Module — re-ranks retrieval results via Cohere API.
Handles list/dict content from Gemini embeddings.
"""

from __future__ import annotations
import logging
import json
from typing import Any, Dict, List, Optional, Union

import cohere

logger = logging.getLogger(__name__)

_client: Optional[cohere.ClientV2] = None


def _get_client() -> cohere.ClientV2:
    """Return cached Cohere client; create on first access."""
    global _client
    if _client is None:
        import os
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise RuntimeError("COHERE_API_KEY is not set. Add it to your .env file.")
        _client = cohere.ClientV2(api_key=api_key)
    return _client


def _extract_text_for_rerank(content: Any) -> str:
    """
    Extract clean text string for Cohere reranking.
    Handles list/dict/JSON structures from Gemini embeddings.
    """
    if content is None:
        return ""
    
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                for key in ["content", "text", "table_data", "llm_generated_caption", "docling_caption"]:
                    if key in item and isinstance(item[key], str):
                        parts.append(item[key])
            elif item is not None:
                parts.append(str(item))
        return " ".join(parts)
    
    if isinstance(content, dict):
        parts = []
        if content.get("type") == "table":
            parts.extend([
                content.get("section_header", ""),
                content.get("docling_caption", ""),
                content.get("llm_generated_caption", ""),
                content.get("table_data", ""),
            ])
        elif content.get("type") == "image":
            parts.extend([
                content.get("section_header", ""),
                content.get("docling_caption", ""),
                content.get("llm_generated_caption", ""),
            ])
        else:
            for key in ["content", "text", "chunk_text", "document", "caption"]:
                if key in content:
                    val = content[key]
                    if isinstance(val, str):
                        parts.append(val)
                    elif isinstance(val, (int, float)):
                        parts.append(str(val))
                    elif isinstance(val, list):
                        parts.extend(str(v) for v in val if v is not None)
        return " ".join(p for p in parts if p and isinstance(p, str))
    
    if isinstance(content, str):
        content = content.strip()
        if not content:
            return ""
        # Try parsing JSON
        if content.startswith("{") or content.startswith("["):
            try:
                parsed = json.loads(content)
                return _extract_text_for_rerank(parsed)
            except (json.JSONDecodeError, TypeError):
                pass
        return content
    
    return str(content)


def rerank_chunks(
    query: str,
    documents: List[Any],  # Can be str, list, or dict
    original_chunks: List[Dict[str, Any]],
    top_n: Optional[int] = None,
    model: str = "rerank-v3.5",
) -> List[Dict[str, Any]]:
    """
    Rerank chunks against query using Cohere Rerank API.
    
    Parameters:
        query: User's natural-language question
        documents: List of chunk contents (str/list/dict from PGVector)
        original_chunks: List of full chunk dicts with ingestion metadata
        top_n: Return only top N results (None = return all)
        model: Cohere rerank model identifier
    
    Returns:
        Reranked list of original chunk dicts, enriched with rerank metadata
    """
    if not documents or not query or not query.strip():
        return original_chunks
    
    # Extract clean text strings for Cohere API
    documents_text = [_extract_text_for_rerank(doc) for doc in documents]
    # Filter out empty strings
    valid_pairs = [(i, txt) for i, txt in enumerate(documents_text) if txt.strip()]
    
    if not valid_pairs:
        return original_chunks
    
    client = _get_client()
    valid_indices = [i for i, _ in valid_pairs]
    valid_texts = [txt for _, txt in valid_pairs]
    
    logger.info("Reranking %d chunks (model=%s, top_n=%s)", len(valid_texts), model, top_n)
    
    try:
        response = client.rerank(
            model=model,
            query=query,
            documents=valid_texts,
            top_n=top_n,
        )
    except Exception as e:
        logger.warning("Cohere rerank API call failed: %s", e)
        return original_chunks
    
    # Rebuild chunks with rerank metadata, preserving original structure
    reranked: List[Dict[str, Any]] = []
    for result in response.results:
        original_idx = valid_indices[result.index] if result.index < len(valid_indices) else None
        if original_idx is not None and 0 <= original_idx < len(original_chunks):
            enriched = {**original_chunks[original_idx]}
            enriched["rerank_score"] = round(result.relevance_score, 6)
            enriched["rerank_index"] = original_idx
            reranked.append(enriched)
    
    logger.info(
        "Reranking complete — returned %d chunks (best score=%.4f)",
        len(reranked),
        reranked[0]["rerank_score"] if reranked else 0.0,
    )
    return reranked