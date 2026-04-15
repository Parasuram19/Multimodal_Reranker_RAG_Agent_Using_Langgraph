"""
sql_agent.py
==============
Standalone agentic SQL assistant for RIL financial data.
Uses LangChain SQLDatabase + Google Gemini to convert natural language
questions into SQL queries, execute them, and return answers.

Returns both the synthesized answer AND the SQL query used.

Usage:
    uv run sql_agent.py "what was RIL gross revenue in Q2 FY25"
    uv run sql_agent.py "compare Jio EBITDA margin across all quarters"
    uv run sql_agent.py "show all operational KPIs for Jio"

Environment:
    DATABASE_URL  — PostgreSQL connection (rag_readonly role)
    GEMINI_API_KEY — Google Gemini API key
"""

import os
import sys
import json
from typing import Optional

from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(override=True)

# ── Config ──────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("ADMIN_DB_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-05-20")

if not DATABASE_URL:
    print("ERROR: DATABASE_URL is not set. Add it to your .env file.")
    sys.exit(1)
if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY is not set. Add it to your .env file.")
    sys.exit(1)

# Tables this agent can query
TABLE_NAMES = [
    "business_segments",
    "consolidated_financials",
    "jpl_financials",
    "jpl_operations",
]


def get_db_schema() -> str:
    """Fetch full schema (DDL + 3 sample rows) directly via psycopg2.
    Replaces SQLDatabase.get_table_info() which fails with rag_readonly.
    """
    conn = psycopg2.connect(DATABASE_URL)
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
                from decimal import Decimal
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


# ── Cache schema at startup ─────────────────────────────────────────────────
print("  Loading database schema...")
DB_SCHEMA = get_db_schema()
print(f"  Schema loaded ({len(TABLE_NAMES)} tables).\n")

# ── LLM ────────────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=GEMINI_API_KEY,
    temperature=0,
)


# ── Prompts ────────────────────────────────────────────────────────────────
QUERY_WRITER_PROMPT = """You are a SQL expert for RELIANCE INDUSTRIES (RIL) financial data.

## Database Schema
{schema}

## Instructions
1. Write a SINGLE correct PostgreSQL SELECT query to answer the user's question.
2. Use ONLY the tables and columns shown in the schema above.
3. ALL monetary values in consolidated_financials and jpl_financials are stored
   in the `value_in_cr` column (INR Crore). Ratios and percentages are also in
   `value_in_cr` but the `unit` column will say 'percentage' or 'ratio'.
4. Periods are stored as strings: '2Q FY25', '1Q FY25', '2Q FY24', '1H FY25', '1H FY24', 'FY24'.
5. For YoY comparisons, compare the same metric across different periods.
6. Use ORDER BY for ranking questions.
7. Use LIMIT 20 to keep results manageable.
8. Return ONLY the SQL query. No explanations, no markdown fences, no backticks.

## User Question
{question}

## SQL Query"""


ANSWER_SYNTHESIZER_PROMPT = """You are a RELIANCE INDUSTRIES financial analyst assistant.

You will receive:
- A user's question
- The SQL query that was executed
- The raw query results

Provide a clear, professional answer. Rules:
1. Present numbers with proper formatting — use INR Crore notation (e.g., ₹258,027 Crore).
2. Always mention the time period (e.g., "in 2Q FY25").
3. For YoY comparisons, state both values and the percentage change.
4. Use bullet points or a mini-table for multi-row results.
5. Add brief analytical context (e.g., "EBITDA declined 2% YoY due to weaker O2C margins").
6. If the result has a `notes` column, incorporate that context.
7. If the query returned no rows, say the data isn't available.

## User Question
{question}

## SQL Query Used
{sql_query}

## Query Results
{results}

## Answer"""


# ── Functions ───────────────────────────────────────────────────────────────

def write_sql_query(question: str) -> str:
    """Use the LLM to convert a natural language question to SQL."""
    prompt = ChatPromptTemplate.from_template(QUERY_WRITER_PROMPT)

    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"schema": DB_SCHEMA, "question": question})

    # Strip markdown fences if the LLM wraps them
    sql = raw.strip()
    if sql.startswith("```"):
        lines = sql.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        sql = "\n".join(lines).strip()

    return sql


def execute_query(sql: str) -> tuple:
    """Execute SQL directly via psycopg2 and return (columns, rows)."""
    from decimal import Decimal

    conn = psycopg2.connect(DATABASE_URL)
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(sql)

        rows_raw = cur.fetchall()

        if not rows_raw:
            return [], []

        columns = list(rows_raw[0].keys())
        rows = [dict(row) for row in rows_raw]

        for row in rows:
            for k, v in row.items():
                if isinstance(v, Decimal):
                    row[k] = float(v)

        return columns, rows
    finally:
        conn.close()


def synthesize_answer(question: str, sql: str, columns: list, rows: list) -> str:
    """Use the LLM to turn query results into a natural language answer."""
    if not rows:
        return "No data found for this query."

    # Format results as a readable table
    header = " | ".join(columns)
    separator = "-+-".join(["---"] * len(columns))
    row_lines = []
    for row in rows[:30]:
        vals = [str(row.get(c, "")) for c in columns]
        row_lines.append(" | ".join(vals))
    if len(rows) > 30:
        row_lines.append(f"... ({len(rows) - 30} more rows)")

    results_text = f"{header}\n{separator}\n" + "\n".join(row_lines)

    prompt = ChatPromptTemplate.from_template(ANSWER_SYNTHESIZER_PROMPT)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "question": question,
        "sql_query": sql,
        "results": results_text,
    })

    return answer.strip()


def ask(question: str, verbose: bool = True) -> dict:
    """
    Main entry point. Ask a financial question, get an agentic answer.

    Args:
        question: Natural language question about RIL financial data.
        verbose: Print intermediate steps to console.

    Returns:
        dict with keys:
          - "question": the original question
          - "sql_query": the SQL that was executed
          - "columns": column names from the result
          - "rows": raw row data
          - "row_count": number of rows returned
          - "answer": the LLM-synthesized natural language answer
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"  Question: {question}")
        print(f"{'='*70}\n")

    # Step 1: Generate SQL
    if verbose:
        print("  [1/3] Generating SQL query...")
    sql = write_sql_query(question)
    if verbose:
        print(f"  SQL: {sql}\n")

    # Step 2: Execute
    if verbose:
        print("  [2/3] Executing query...")
    columns, rows = execute_query(sql)
    if verbose:
        print(f"  Returned {len(rows)} rows.\n")

    # Step 3: Synthesize answer
    if verbose:
        print("  [3/3] Synthesizing answer...\n")
    answer = synthesize_answer(question, sql, columns, rows)

    result = {
        "question": question,
        "sql_query": sql,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "answer": answer,
    }

    return result


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: uv run sql_agent.py \"your question here\"")
        print()
        print("Examples:")
        print('  uv run sql_agent.py "what was RIL gross revenue in Q2 FY25"')
        print('  uv run sql_agent.py "compare Jio EBITDA margin across all quarters"')
        print('  uv run sql_agent.py "show all Jio operational KPIs"')
        print('  uv run sql_agent.py "what is the net debt to EBITDA ratio"')
        print('  uv run sql_agent.py "rank all profitability metrics for 2Q FY25"')
        print()
        print("Interactive mode (no argument):")
        print("  uv run sql_agent.py")
        sys.exit(0)

    # Single query mode
    question = " ".join(sys.argv[1:])
    result = ask(question)

    # Print answer
    print(f"{'─'*70}")
    print(f"  ANSWER")
    print(f"{'─'*70}")
    print(f"\n{result['answer']}\n")

    # Print SQL used
    print(f"{'─'*70}")
    print(f"  SQL QUERY USED")
    print(f"{'─'*70}")
    print(f"\n{result['sql_query']}\n")

    # Print raw data
    print(f"{'─'*70}")
    print(f"  RAW DATA ({result['row_count']} rows)")
    print(f"{'─'*70}")
    if result["columns"] and result["rows"]:
        header = " | ".join(result["columns"])
        print(f"\n{header}")
        print("-+-".join(["---"] * len(result["columns"])))
        for row in result["rows"][:20]:
            vals = [str(row.get(c, "")) for c in result["columns"]]
            print(" | ".join(vals))
        if len(result["rows"]) > 20:
            print(f"... ({len(result['rows']) - 20} more rows)")
    else:
        print("\n  (no data)")
    print()

    # Return as JSON for programmatic use
    # Strip raw rows from console but keep in return value
    return result


def interactive():
    """REPL mode — keep asking questions until the user quits."""
    print()
    print("=" * 70)
    print("  RIL Financial SQL Agent — Interactive Mode")
    print("  Type your question and press Enter. Type 'quit' or 'exit' to stop.")
    print("=" * 70)

    history = []

    while True:
        try:
            question = input("\n  Ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Bye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("\n  Bye!")
            break

        result = ask(question, verbose=True)

        print(f"\n  {'─'*60}")
        print(f"  {result['answer']}")
        print(f"  {'─'*60}")
        print(f"  SQL: {result['sql_query']}")
        print(f"  Rows: {result['row_count']}")

        history.append(result)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        interactive()
    else:
        main()
