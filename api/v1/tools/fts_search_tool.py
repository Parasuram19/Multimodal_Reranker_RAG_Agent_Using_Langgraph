"""
Full-Text Search Tool — keyword search using PostgreSQL tsvector.
Returns chunks with content normalized to string.
"""

import os
import json
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv(override=True)

_RAW_CONN = os.getenv("PG_STR")
_COLLECTION_NAME = "financial_rag"

_FTS_SQL = """
    SELECT
        e.document                                               AS content,
        e.cmetadata                                              AS metadata,
        ts_rank(
            to_tsvector('english', e.document),
            plainto_tsquery('english', %(query)s)
        )                                                        AS fts_rank
    FROM langchain_pg_embedding e
    JOIN langchain_pg_collection c ON c.uuid = e.collection_id
    WHERE c.name = %(collection)s
      AND to_tsvector('english', e.document)
          @@ plainto_tsquery('english', %(query)s)
    ORDER BY fts_rank DESC
    LIMIT %(k)s;
"""


@tool
def fts_search(query: str, k: int = 5) -> list[dict]:
    """
    Perform full-text (keyword) search against the Policy knowledge base.
    Returns chunks with content normalized to string.
    """
    print(f"[fts_search] query='{query}', k={k}")
    
    if not _RAW_CONN:
        print("[fts_search] ERROR: PG_CONNECTION_STRING not set")
        return []
    
    with psycopg.connect(_RAW_CONN, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                _FTS_SQL,
                {"query": query, "collection": _COLLECTION_NAME, "k": k},
            )
            rows = cur.fetchall()
    
    results = []
    for row in rows:
        # Normalize content to string
        content = row["content"]
        if isinstance(content, (list, dict)):
            content = json.dumps(content, ensure_ascii=False)
        elif content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)
        
        # Ensure metadata is a dict
        metadata = row["metadata"] if isinstance(row["metadata"], dict) else {}
        
        chunk = {
            "content": content,
            "metadata": metadata,
            "fts_rank": round(float(row["fts_rank"]), 4),
        }
        results.append(chunk)
    
    print(f"[fts_search] returned {len(results)} chunks")
    return results