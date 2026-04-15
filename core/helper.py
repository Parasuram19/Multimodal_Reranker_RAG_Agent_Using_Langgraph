"""
Core helper functions for vector store, embeddings, and SQL database access.
"""

import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.utilities import SQLDatabase

load_dotenv(override=True)

PG_CONNECTION = os.getenv("DATABASE_URL")
if not PG_CONNECTION:
    raise ValueError("PG_CONNECTION_STRING or ADMIN_DB_URL is not set")

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set")

EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDINGS_MODEL", "gemini-embedding-001")


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
        collection_name: Name of the PGVector collection (default: "financial_rag")

    Returns:
        PGVector instance configured with Gemini embeddings
    """
    return PGVector(
        collection_name=collection_name,
        connection=PG_CONNECTION,
        embeddings=get_embedding_model(),
        use_jsonb=True  # Store metadata as JSONB for flexible querying
    )


def get_sql_database() -> SQLDatabase:
    """Return a LangChain SQLDatabase connected to the financial database (read-only).

    Uses the rag_readonly role from sql/seed.sql with SELECT privileges only.
    Includes all financial tables for the NL2SQL agent.
    """
    db_url = os.getenv("ADMIN_DB_URL")
    if not db_url:
        raise ValueError("ADMIN_DB_URL is not set. Check your .env file.")

    return SQLDatabase.from_uri(
        db_url,
        include_tables=[
            "jio_users",
            "reliance_shareholders",
        ],
        sample_rows_in_table_info=3,
    )


def get_db_schema() -> str:
    """Fetch full schema (DDL + 3 sample rows) directly via psycopg2.

    This bypasses SQLDatabase.get_table_info() which may fail with
    certain read-only roles or permission setups.
    """
    import psycopg2
    from decimal import Decimal

    db_url = os.getenv("ADMIN_DB_URL")
    if not db_url:
        raise ValueError("ADMIN_DB_URL is not set. Check your .env file.")

    TABLE_NAMES = [
            "jio_users",
            "reliance_shareholders",
        ]

    conn = psycopg2.connect(db_url)
    try:
        cur = conn.cursor()
        schema_parts = []

        for table in TABLE_NAMES:
            # Get column definitions
            cur.execute("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position;
            """, (table,))
            columns = cur.fetchall()

            ddl_lines = [f"TABLE {table} ("]
            col_defs = []
            for col_name, data_type, nullable, default in columns:
                d = f"  {col_name}  {data_type}"
                if nullable == "NO":
                    d += "  NOT NULL"
                if default is not None:
                    d += f"  DEFAULT {default}"
                col_defs.append(d)
            ddl_lines.append(",\n".join(col_defs))
            ddl_lines.append(");\n")

            # Get 3 sample rows
            cur.execute(f'SELECT * FROM "{table}" LIMIT 3;')
            rows = cur.fetchall()
            col_names = [desc[0] for desc in cur.description]

            sample_lines = []
            for row in rows:
                vals = []
                for v in row:
                    if isinstance(v, Decimal):
                        vals.append(str(float(v)))
                    elif isinstance(v, str) and len(v) > 100:
                        vals.append(v[:100] + "...")
                    else:
                        vals.append(str(v) if v is not None else "NULL")
                sample_lines.append(f"  ({', '.join(vals)})")

            schema_parts.append(
                "\n".join(ddl_lines)
                + f"/*\nSample rows:\n" + "\n".join(sample_lines) + "\n*/"
            )

        return "\n\n".join(schema_parts)
    finally:
        conn.close()