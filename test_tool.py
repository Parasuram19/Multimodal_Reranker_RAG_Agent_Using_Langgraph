"""
🔍 Debug Script: Fetch & Inspect Chunks from Tools
"""
import os, sys, json
from dotenv import load_dotenv
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
load_dotenv(override=True)

from api.v1.tools.vector_search_tool import vector_search
from api.v1.tools.fts_search_tool import fts_search
from api.v1.tools.hybrid_search_tool import hybrid_search

def print_chunks(chunks, tool_name):
    print(f"\n{'='*50}")
    print(f"📂 {tool_name.upper()} RESULTS ({len(chunks)} chunks)")
    print(f"{'='*50}")
    
    if not chunks:
        print("❌ No chunks returned.")
        return

    for i, chunk in enumerate(chunks, 1):
        content = chunk.get("content", "")
        # Truncate long content
        display_content = str(content)[:200] + "..." if len(str(content)) > 200 else str(content)
        
        print(f"\n🔹 Chunk {i}:")
        print(f"   Content: {display_content}")
        
        meta = chunk.get("metadata", {})
        if isinstance(meta, dict):
            # Print key metadata fields
            for key in ["modality", "document_name", "page_number", "section_header"]:
                if meta.get(key):
                    print(f"   [{key}]: {meta[key]}")
        
        if "fts_rank" in chunk:
            print(f"   📊 FTS Rank: {chunk['fts_rank']}")
        if "hybrid_score" in chunk.get("metadata", {}):
            print(f"   📊 Hybrid Score: {chunk['metadata']['hybrid_score']}")

if __name__ == "__main__":
    test_queries = ["EBITDA?", "Lower gas price"]
    
    for q in test_queries:
        print(f"\n🔍 TESTING QUERY: '{q}'")
        
        # 1. Vector
        try:
            v_res = vector_search.func(query=q, k=3)
            print_chunks(v_res, "Vector Search")
        except Exception as e:
            print(f"❌ Vector Error: {e}")
            
        # 2. FTS
        try:
            f_res = fts_search.func(query=q, k=3)
            print_chunks(f_res, "FTS Search")
        except Exception as e:
            print(f"❌ FTS Error: {e}")
            
        # 3. Hybrid
        try:
            h_res = hybrid_search.func(query=q, k=3)
            print_chunks(h_res, "Hybrid Search")
        except Exception as e:
            print(f"❌ Hybrid Error: {e}")