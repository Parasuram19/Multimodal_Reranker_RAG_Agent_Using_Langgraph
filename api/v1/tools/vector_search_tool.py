"""
Vector Search Tool — semantic search against Policy knowledge base.
Returns chunks with content as string (not list) for compatibility.
"""

from langchain.tools import tool
from core.helper import get_vector_store


@tool
def vector_search(query: str, k: int = 5) -> list[dict]:
    """
    Perform semantic vector search against the Policy knowledge base.
    
    Returns chunks with content normalized to string format.
    """
    print(f"[vector_search] query='{query}', k={k}")
    
    vector_store = get_vector_store()
    docs = vector_store.similarity_search(query, k=k)
    
    results = []
    for doc in docs:
        # Normalize content to string regardless of storage format
        content = doc.page_content
        if isinstance(content, (list, dict)):
            import json
            content = json.dumps(content, ensure_ascii=False)
        elif content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)
        
        # Ensure metadata is a dict
        metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
        
        chunk = {
            "content": content,  # Always a string
            "metadata": metadata,
        }
        results.append(chunk)
    
    print(f"[vector_search] returned {len(results)} chunks")
    return results