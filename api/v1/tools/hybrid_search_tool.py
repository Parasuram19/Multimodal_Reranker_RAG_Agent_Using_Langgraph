"""
Hybrid Search Tool — combines vector + FTS using Reciprocal Rank Fusion.
Returns chunks with content normalized to string.
"""

import json
from langchain.tools import tool
from core.helper import get_vector_store
from api.v1.tools.fts_search_tool import fts_search
from typing import Any
_RRF_K = 60


@tool
def hybrid_search(query: str, k: int = 5) -> list[dict]:
    """
    Perform hybrid search (vector + full-text) using Reciprocal Rank Fusion.
    Returns chunks with content normalized to string.
    """
    print(f"[hybrid_search] query='{query}', k={k}")
    
    vector_store = get_vector_store()
    
    # Get results from both methods
    vector_docs = vector_store.similarity_search(query, k=k)
    fts_docs = fts_search.func(query=query, k=k)
    
    # Reciprocal Rank Fusion
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}
    
    def _make_key(content: Any, meta: dict) -> str:
        """Create a unique key for RRF deduplication."""
        # Normalize content to string for key generation
        if isinstance(content, (list, dict)):
            content_str = json.dumps(content, sort_keys=True, ensure_ascii=False)[:100]
        else:
            content_str = str(content)[:100]
        page = meta.get("page_number") or meta.get("page", "")
        return f"{content_str}_{page}"
    
    # Process vector results
    for rank, doc in enumerate(vector_docs):
        content = doc.page_content
        if isinstance(content, (list, dict)):
            content = json.dumps(content, ensure_ascii=False)
        elif content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)
        
        key = _make_key(content, doc.metadata)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
        chunk_map[key] = {
            "content": content,
            "metadata": doc.metadata if isinstance(doc.metadata, dict) else {},
        }
    
    # Process FTS results
    for rank, item in enumerate(fts_docs):
        content = item["content"]
        if isinstance(content, (list, dict)):
            content = json.dumps(content, ensure_ascii=False)
        elif content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)
        
        key = _make_key(content, item["metadata"])
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
        chunk_map[key] = {
            "content": content,
            "metadata": item["metadata"] if isinstance(item["metadata"], dict) else {},
        }
    
    # Sort by RRF score and return top-k
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for key, _ in ranked[:k]:
        chunk = chunk_map[key]
        chunk["metadata"]["hybrid_score"] = round(rrf_scores[key], 4)
        results.append(chunk)
    
    print(f"[hybrid_search] returned {len(results)} fused chunks")
    return results