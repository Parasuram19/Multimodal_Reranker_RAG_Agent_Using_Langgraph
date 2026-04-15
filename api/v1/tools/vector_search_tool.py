"""
Vector Search Tool — semantic search against Policy knowledge base.
Uses Parent-Child Retrieval: Searches embedded children, returns full parent context.
"""

import json
from langchain.tools import tool
from core.helper import get_vector_store

@tool
def vector_search(query: str, k: int = 20) -> list[dict]:
    """
    Perform semantic vector search against the Policy knowledge base.
    
    Returns full parent chunks with content normalized to string format.
    """
    print(f"[vector_search] query='{query}', k={k}")
    
    vector_store = get_vector_store()
    
    # 1. Embed the query
    query_embedding = vector_store.embeddings.embed_query(query)
    
    # 2. Search ONLY child chunks (fetch extra to find unique parents)
    child_docs = vector_store.similarity_search_by_vector(
        embedding=query_embedding,
        k=k * 3, 
        filter={"is_parent": False}
    )
    
    # 3. Extract unique parent IDs (preserving rank order)
    parent_ids = []
    for doc in child_docs:
        pid = doc.metadata.get("parent_chunk_id")
        if pid and pid not in parent_ids:
            parent_ids.append(pid)
            
    # Cap the number of parents to return to avoid context window bloat
    parent_ids = parent_ids[:k]
    
    if not parent_ids:
        return []
        
    # 4. Fetch the full parent documents
    parent_docs = vector_store.get_by_ids(parent_ids)
    
    # Map parents to preserve the ranked order of the children that found them
    parent_map = {}
    for doc in parent_docs:
        doc_id = doc.metadata.get("document_id")
        c_idx = doc.metadata.get("chunk_index")
        pid = f"{doc_id}_p{c_idx}"
        parent_map[pid] = doc
        
    # 5. Format results
    results = []
    for pid in parent_ids:
        if pid in parent_map:
            doc = parent_map[pid]
            content = doc.page_content
            
            # Normalize content to string
            if isinstance(content, (list, dict)):
                content = json.dumps(content, ensure_ascii=False)
            elif content is None:
                content = ""
            elif not isinstance(content, str):
                content = str(content)
                
            metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
            
            results.append({
                "content": content,
                "metadata": metadata,
            })
            
    print(f"[vector_search] returned {len(results)} parent chunks")
    return results