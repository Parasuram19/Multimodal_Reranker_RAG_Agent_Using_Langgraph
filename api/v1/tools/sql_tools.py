# api/v1/tools/sql_tools.py
"""
SQL database tools for the agentic RAG pipeline.

Provides two tools:
  - sql_schema_inspector: Returns schema info for available tables.
  - sql_query_executor:   Executes a read-only SELECT query and returns rows.

Connection is read-only (rag_readonly role). Only SELECT statements are allowed.
"""

import json
import logging
import os
from typing import List

from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def get_sql_database() -> SQLDatabase:
    """Return a LangChain SQLDatabase connected to agentic_rag_db (read-only).

    Uses the rag_readonly role from sql/seed.sql — SELECT privileges only.
    Connection string is read from ADMIN_DB_URL in the environment.
    """
    db_url = os.getenv("ADMIN_DB_URL")
    if not db_url:
        raise ValueError(
            "ADMIN_DB_URL is not set. Check your .env file."
        )
    return SQLDatabase.from_uri(
        db_url,
        include_tables=[
            "business_segments",
            "consolidated_financials",
            "jpl_financials",
            "jpl_operations",
        ],
        sample_rows_in_table_info=2,
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def sql_schema_inspector(table_names: str = "") -> str:
    """
    Return the CREATE TABLE DDL and sample rows for the requested tables
    in the agentic_rag_db.

    Use this FIRST whenever you need to write a SQL query — it tells you the
    exact column names, types, and relationships.

    Args:
        table_names: Comma-separated list of table names to inspect.
                     If empty, returns schema for ALL available tables.
                     Available: business_segments, consolidated_financials,
                     jpl_financials, jpl_operations

    Returns:
        JSON string with schema and sample rows for each table.
    """
    try:
        db = get_sql_database()
        all_tables = db.get_usable_table_names()

        if table_names.strip():
            requested = [
                t.strip() for t in table_names.split(",") if t.strip()
            ]
            valid = [t for t in requested if t in all_tables]
            invalid = set(requested) - set(valid)
            if invalid:
                return json.dumps({
                    "error": f"Unknown tables: {invalid}",
                    "available_tables": all_tables,
                })
            tables_to_inspect = valid
        else:
            tables_to_inspect = all_tables

        schema_info = {}
        for table in tables_to_inspect:
            try:
                info = db.get_table_info(table_names=[table])
                schema_info[table] = info
            except Exception as exc:
                logger.warning(
                    "sql_schema_inspector: failed for table %s: %s",
                    table, exc,
                )
                schema_info[table] = f"Error retrieving schema: {exc}"

        return json.dumps(
            {"tables": schema_info, "available_tables": all_tables},
            indent=2,
        )

    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as exc:
        logger.error("sql_schema_inspector: unexpected error: %s", exc)
        return json.dumps({"error": f"Database connection failed: {exc}"})


@tool
def sql_query_executor(sql_query: str) -> str:
    """
    Execute a read-only SELECT query against the agentic_rag_db and return
    the results as a JSON array of row objects.

    ONLY SELECT statements are allowed. INSERT, UPDATE, DELETE, DROP, ALTER,
    CREATE, TRUNCATE will be rejected.

    Args:
        sql_query: A valid SQL SELECT query. Example:
            "SELECT metric_name, value_in_cr
             FROM consolidated_financials
             WHERE period = '2Q FY25'
             ORDER BY value_in_cr DESC LIMIT 10;"

    Returns:
        JSON with columns list and rows list, or an error message.
    """
    normalized = sql_query.strip().upper()

    # ---- Safety guard ----
    forbidden_patterns = [
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
        "TRUNCATE", "GRANT", "REVOKE", "EXECUTE", "CALL",
    ]
    for pattern in forbidden_patterns:
        if pattern in normalized:
            return json.dumps({
                "error": f"Forbidden operation: {pattern}. "
                         "Only SELECT queries are allowed.",
            })

    if not normalized.startswith("SELECT"):
        return json.dumps({
            "error": "Only SELECT queries are allowed. "
                     "Query must start with SELECT.",
        })

    clean_query = sql_query.strip().rstrip(";").strip()

    try:
        db = get_sql_database()
        result = db.run(clean_query)

        if not result or not result.strip():
            return json.dumps({
                "columns": [],
                "rows": [],
                "row_count": 0,
                "message": "Query executed successfully but returned no rows.",
            })

        try:
            import ast
            rows_raw = ast.literal_eval(result.strip())
            if isinstance(rows_raw, list) and rows_raw:
                if isinstance(rows_raw[0], dict):
                    columns = list(rows_raw[0].keys())
                    rows = rows_raw
                else:
                    columns = [
                        f"col_{i}" for i in range(len(rows_raw[0]))
                    ]
                    rows = [
                        dict(zip(columns, row)) for row in rows_raw
                    ]
            else:
                rows = []
                columns = []
        except (ValueError, SyntaxError, TypeError):
            rows = []
            columns = []

        return json.dumps(
            {
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
            },
            indent=2,
            default=str,
        )

    except Exception as exc:
        logger.error("sql_query_executor: %s", exc)
        return json.dumps({
            "error": f"SQL execution failed: {exc}",
            "query": clean_query,
        })


SQL_TOOLS: List = [sql_schema_inspector, sql_query_executor]
