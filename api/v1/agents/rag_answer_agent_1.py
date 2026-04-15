import json
import logging
import os
from typing import Annotated, Any, Dict, List, Optional, TypedDict, Union

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from api.v1.schemas.query_schema import QueryResponse, MetadataEntry
from api.v1.tools.fts_search_tool import fts_search
from api.v1.tools.hybrid_search_tool import hybrid_search
from api.v1.tools.vector_search_tool import vector_search
from api.v1.tools.sql_tools import sql_schema_inspector, sql_query_executor, SQL_TOOLS
from api.v1.utils.reranker import rerank_chunks
from api.v1.utils.scoring import score_chunks

load_dotenv(override=True)
logger = logging.getLogger(__name__)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Config                                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
ENABLE_RERANK: bool = os.getenv("ENABLE_RERANK", "true").lower() == "true"
RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", "10"))
RERANK_MODEL: str = os.getenv("RERANK_MODEL", "rerank-v3.5")
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  LLM                                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is required")

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)

# RAG retrieval tools
RETRIEVAL_TOOLS: List[BaseTool] = [vector_search, fts_search, hybrid_search]
llm_with_retrieval_tools = llm.bind_tools(RETRIEVAL_TOOLS)

# SQL tools — bound to a separate LLM instance for the SQL path
llm_with_sql_tools = llm.bind_tools(SQL_TOOLS)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Prompts                                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ---- NEW: Query Router ----
QUERY_ROUTER_PROMPT = """You are a query router for a financial assistant that has TWO data sources:

1. **SQL Database** — structured data in tables: products, categories, orders, order_items.
   Good for: exact counts, aggregations, listings, comparisons of structured records,
   "how many", "top N", "most expensive", "total revenue", order details, product catalogs,
   category distributions, price ranges, stock quantities.

2. **Document Store (RAG)** — unstructured financial documents (PDFs, reports).
   Good for: narrative analysis, explanations, trends, qualitative information,
   "what does the report say about X", "analyze the strategy", "risk factors",
   "management discussion", any question that requires reading document content.

Classify the user query into one of these routes:

- **"sql"** — The query asks about structured data that likely lives in a database.
  Examples: "how many orders were placed", "top 5 products by revenue",
  "list all categories", "total sales last month", "most expensive product".

- **"rag"** — The query asks about information that would be found in documents.
  Examples: "what does the annual report say about growth strategy",
  "analyze the risk factors mentioned in the Q2 filing",
  "summarize the management discussion", "what are the key financial highlights".

- **"both"** — The query needs BOTH database data AND document context.
  Examples: "compare product revenue with what the report projected",
  "are our order trends aligned with the strategy document",
  "what does the report say about our top-selling products".

Return ONLY a JSON object: {{"route": "sql"|"rag"|"both", "reason": "one sentence"}}"""


# ---- NEW: SQL Agent ----
SQL_AGENT_SYSTEM_PROMPT = """You are a SQL expert assistant. You have access to a relational database
with these tables: products, categories, orders, order_items.

You have two tools:
  - sql_schema_inspector: Get table DDL and sample rows. USE THIS FIRST.
  - sql_query_executor: Execute a SELECT query.

Rules:
1. ALWAYS call sql_schema_inspector first to understand the exact column names and types.
2. Then write a precise SELECT query and call sql_query_executor.
3. ONLY call sql_schema_inspector once per turn — you already know the schema.
4. Do NOT answer the question yourself — return ONLY tool calls.
5. Write efficient queries: use JOINs, LIMIT, ORDER BY as appropriate.
6. NEVER call both tools in the same response — inspect schema first, then query."""


# ---- NEW: SQL Synthesis (goes directly to synthesize, no rerank/score) ----
SQL_SYNTHESIZE_SYSTEM_PROMPT = """You are a financial data assistant. You have SQL query results
from a structured database (products, categories, orders, order_items).

You will be given:
- A user question
- SQL query results (rows/columns from the database)

Provide the BEST possible answer using the SQL results.

Rules:
1. Present data in a clear, structured way — use tables or bullet points for numeric data.
2. Include specific numbers, percentages, and comparisons when available.
3. If the query returned no rows, say so clearly and suggest what data might be needed.
4. If the user asked for a ranking or top-N, present it as a ranked list.
5. Add brief context or insights when possible (e.g., "Product X accounts for 35% of total revenue").

Return your answer as a JSON object with exactly these keys:
  - "answer": your response (string)
  - "metadata": []

Return ONLY the JSON object. Do NOT wrap it in markdown code fences."""


# ---- NEW: Combined Synthesis (RAG + SQL) ----
COMBINED_SYNTHESIZE_SYSTEM_PROMPT = """You are a financial data assistant. You have TWO sources of
information to answer the user's question:

1. **SQL Database Results** — structured data from products, categories, orders, order_items.
2. **Document Chunks** — relevant excerpts from financial documents.

You will be given:
- A user question
- SQL query results (if available)
- Document chunks (if available)

Provide the BEST possible answer by combining insights from BOTH sources.

Rules:
1. Start with the SQL data (it's precise and factual), then enrich with document context.
2. If the two sources seem to disagree, note the discrepancy.
3. Cite both sources: "According to our database..." and "The report states...".
4. Present structured data (numbers, rankings) clearly.
5. Add analytical insights by cross-referencing the two sources.

Return your answer as a JSON object with exactly these keys:
  - "answer": your response (string)
  - "metadata": []

Return ONLY the JSON object. Do NOT wrap it in markdown code fences."""


# ---- Existing prompts (unchanged) ----
AGENT_SYSTEM_PROMPT = """You are an expert Financial Document assistant.

You have access to three retrieval tools:
  • vector_search  → best for natural language / conceptual questions
  • fts_search     → best for codes, IDs, abbreviations, exact keywords  
  • hybrid_search  → best for short or ambiguous queries

Rules:
1. Choose exactly ONE tool based on the query type.
2. Call the tool with the ORIGINAL user query (do not modify it).
3. Return ONLY a tool call — do not answer the question yourself.
4. The tool will return document chunks with metadata including:
   - modality: "text", "table", or "image"
   - section_header: the document section this chunk belongs to
   - For tables: table_data, docling_caption, llm_generated_caption
   - For images: llm_generated_caption describing the chart/figure
5. After retrieval, you will synthesize an answer using ONLY the returned chunks."""

AGENT_RETRY_SYSTEM_PROMPT = """You are an expert Financial Document assistant.

The previous retrieval attempt returned NO relevant document chunks.

You have access to three retrieval tools:
  • vector_search  → best for natural language / conceptual questions
  • fts_search     → best for codes, IDs, abbreviations, exact keywords
  • hybrid_search  → best for short or ambiguous queries

Rules:
1. Choose exactly ONE tool and call it with the REFORMULATED query 
   provided in the most recent human message.
2. You MUST return a tool call — do not answer the question yourself.
3. The reformulated query is optimized for document retrieval."""

QUERY_REFORMULATE_SYSTEM_PROMPT = """You rewrite user questions for better document retrieval in a financial report system.

Your job:
1. Read the user's original question.
2. Express what the user is already asking — do NOT change the core meaning or intent.
3. Rewrite it as a natural sentence that sounds like something a financial report would discuss.
4. If the question has multiple aspects, split them into separate short keyword-style queries joined by " OR ".
5. Do NOT add a rigid structure or template. Just say it plainly in report language.

Examples:
  User: "how much was spent on marketing"
  → "marketing expenditure OR advertising spend OR promotional costs"

  User: "what is the consumer base growth"
  → "customer base growth OR subscriber growth OR user metrics trend"

  User: "tell me about profit"
  → "profit OR EBITDA OR net income OR PAT"

  User: "what happened with revenue in Q2"
  → "Q2 revenue OR second quarter top-line OR Q2 FY turnover"

Return ONLY the reformulated query. Nothing else."""

SYNTHESIZE_SYSTEM_PROMPT = """You are an expert financial document assistant.

You will be given document chunks and a user question. Provide the BEST
possible answer using the available chunks.

## Rules

1. DIRECT MATCH: If a chunk directly answers the question, use it.

2. PARTIAL / RELATED MATCH (CRITICAL): If no chunk has the exact answer
   BUT contains RELATED data, you MUST provide that related data and
   explain how it connects. Do NOT say "no information found" when
   related metrics exist. Examples:
   - User asks "amount spent on consumers" → provide ARPU, customer count,
     revenue per user from available tables
   - User asks "operation cost" → provide operational metrics, OpEx data,
     or the closest available financial metrics
   - User asks "profit" → provide EBITDA, PAT, or margin data

3. FINANCIAL TERM MAPPING — treat these as equivalent:
   - amount spent / cost / expenditure → ARPU, revenue, OpEx
   - consumer base / customer base → subscriber count, user metrics
   - operation cost / operational cost → OPEX, operational metrics
   - income / earnings / profit → Revenue, EBITDA, PAT

4. TABLE DATA: Extract and present specific row/column values from tables.
   Tables often contain the most precise answers — never ignore them.

5. ONLY say "the documents do not contain this information" when ZERO
   chunks are even tangentially related to the question.

Return your answer as a JSON object with exactly these keys:
  - "answer": your synthesized response (string)
  - "metadata": []

Return ONLY the JSON object. Do NOT wrap it in markdown code fences."""

CHUNK_RELEVANCE_PROMPT = """You are checking if retrieved document chunks are relevant to a user's question.

User Question: {query}

Retrieved Chunks:
{chunks_text}

Does ANY chunk contain information that can answer or is directly related to the user's question?

- YES → if at least one chunk is about the same entity, metric, time period, or topic the user asked about.
- NO → if the chunks are about completely different entities, topics, or metrics that do not match what the user asked.

Example:
  Question: "gross profit of JPL in Q2 FY25"
  Chunks about: oil and gas prices, energy sector trends → NO
  Chunks about: JPL financials, JPL profit, JPL Q2 results → YES

Return JSON: {{"relevant": true/false}}"""


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Graph State (updated with SQL fields)                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
class RAGState(TypedDict):
    """State that flows through every node in the graph."""
    messages: Annotated[list, add_messages]
    query: str
    query_route: str                    # NEW: "sql", "rag", or "both"
    original_query: str                 # NEW: preserve the original user query
    raw_chunks: List[Dict[str, Any]]
    reranked_chunks: List[Dict[str, Any]]
    metadata: List[Dict[str, Any]]
    retries: int
    synthesized_answer: str
    answer_relevant: bool

    # ---- NEW: SQL path fields ----
    sql_results: Dict[str, Any]         # Parsed SQL query output (columns, rows)
    sql_schema: str                     # Cached schema string (avoid re-fetching)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Helper Utilities (all existing helpers preserved)                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
TOOL_NAMES = {"vector_search", "fts_search", "hybrid_search"}
SQL_TOOL_NAMES = {"sql_schema_inspector", "sql_query_executor"}


def _extract_response_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        return "".join(parts)

    if isinstance(content, dict) and "text" in content:
        return content["text"]

    return str(content) if content else ""


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.index("\n") if "\n" in text else len(text)
        text = text[first_newline:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        return text.strip()
    return text


def _extract_text_from_chunk(chunk: Dict[str, Any]) -> str:
    content = chunk.get("content")

    if content is None:
        return ""

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                for key in ["content", "text", "table_data",
                            "llm_generated_caption", "docling_caption"]:
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
                if key in content and isinstance(content[key], str):
                    parts.append(content[key])
        return " ".join(p for p in parts if p)

    if isinstance(content, str):
        content = content.strip()
        if not content:
            return ""
        if content.startswith("{") or content.startswith("["):
            try:
                parsed = json.loads(content)
                return _extract_text_from_chunk({"content": parsed})
            except (json.JSONDecodeError, TypeError):
                pass
        return content

    return str(content)


def _get_chunk_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(chunk, dict):
        meta = chunk.get("metadata", {})
        if isinstance(meta, dict) and "metadata" in meta:
            meta = {**meta, **meta["metadata"]}
    else:
        meta = {}

    content = chunk.get("content", "") or ""
    if isinstance(content, str) and content.startswith("{"):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                if parsed.get("llm_generated_caption") and not meta.get("llm_generated_caption"):
                    meta["llm_generated_caption"] = parsed["llm_generated_caption"]
                if parsed.get("table_data") and not meta.get("table_data"):
                    meta["table_data"] = parsed["table_data"]
                if parsed.get("docling_caption") and not meta.get("docling_caption"):
                    meta["docling_caption"] = parsed["docling_caption"]
        except (json.JSONDecodeError, TypeError):
            pass

    normalized = {
        "document_id": meta.get("document_id"),
        "document_name": meta.get("document_name") or meta.get("source_file"),
        "page_number": meta.get("page") or meta.get("page_number"),
        "chunk_index": meta.get("chunk_index"),
        "modality": meta.get("modality", "text"),
        "element_type": meta.get("element_type"),
        "section_header": meta.get("section") or meta.get("section_header"),
        "docling_label": meta.get("docling_label"),
        "docling_caption": meta.get("docling_caption"),
        "llm_generated_caption": meta.get("llm_generated_caption"),
        "table_data": meta.get("table_data"),
        "table_title": meta.get("table_title"),
        "bbox": meta.get("bbox"),
        "source_file": meta.get("source_file"),
        "ingested_at": meta.get("ingested_at"),
        "cosine_similarity": None,
        "bm25": None,
        "relevance_score": None,
        "rerank_score": None,
        "rerank_index": None,
    }

    skip_keys = set(normalized.keys())
    for k, v in meta.items():
        if k not in skip_keys and v is not None:
            normalized[k] = v

    return {k: v for k, v in normalized.items() if v is not None}


def _extract_chunks_from_messages(messages: list) -> List[Dict[str, Any]]:
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.name in TOOL_NAMES:
            try:
                parsed = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                if isinstance(parsed, list):
                    return parsed
                elif isinstance(parsed, dict):
                    return parsed.get("results", [parsed]) if "results" in parsed else [parsed]
            except (json.JSONDecodeError, TypeError, AttributeError):
                if isinstance(msg.content, list):
                    return msg.content
                elif isinstance(msg.content, dict):
                    return [msg.content]
    return []


def _extract_sql_results_from_messages(messages: list) -> Dict[str, Any]:
    """Extract parsed SQL results from the latest sql_query_executor ToolMessage."""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.name == "sql_query_executor":
            try:
                content = msg.content
                parsed = json.loads(content) if isinstance(content, str) else content
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
    return {"columns": [], "rows": [], "row_count": 0}


def _build_metadata_entry(
    chunk: Dict[str, Any],
    rank: int,
    scores: Dict[str, Any],
) -> Dict[str, Any]:
    meta = _get_chunk_metadata(chunk)
    text = _extract_text_from_chunk(chunk)

    entry = {
        "rank": rank,
        **scores,
        "rerank_score": chunk.get("rerank_score"),
        "rerank_index": chunk.get("rerank_index"),
        "citation": meta.get("citation", text[:500]),
        "content": text,
        **{k: v for k, v in meta.items() if k not in {
            "rank", "cosine_similarity", "bm25", "relevance_score",
            "citation", "content", "rerank_score", "rerank_index"
        }},
    }
    return entry


def _format_chunks_as_context(chunks: List[Dict[str, Any]], max_chunks: int = 5) -> str:
    parts = []
    for i, chunk in enumerate(chunks[:max_chunks], start=1):
        text = _extract_text_from_chunk(chunk)
        meta = _get_chunk_metadata(chunk)

        modality = meta.get("modality", "text")
        doc = meta.get("document_name", "unknown")
        page = meta.get("page_number", "N/A")
        section = meta.get("section_header", "")

        if modality == "table":
            caption = (
                meta.get("llm_generated_caption")
                or meta.get("docling_caption")
                or ""
            )
            table_preview = text[:1500] + "..." if len(text) > 1500 else text
            parts.append(
                f"[Table {i} — {doc}, p.{page}, Section: {section}]\n"
                f"Caption: {caption}\n"
                f"{table_preview}"
            )
        elif modality == "image":
            caption = (
                meta.get("llm_generated_caption")
                or meta.get("docling_caption")
                or "No caption"
            )
            parts.append(
                f"[Chart/Figure {i} — {doc}, p.{page}, Section: {section}]\n"
                f"Description: {caption}"
            )
        else:
            text_preview = text[:400] + "..." if len(text) > 400 else text
            parts.append(
                f"[Text {i} — {doc}, p.{page}, Section: {section}]\n"
                f"{text_preview}"
            )

    return "\n\n---\n\n".join(parts)


def _format_sql_results_as_context(sql_results: Dict[str, Any]) -> str:
    """Format SQL query output into a readable context block for synthesis."""
    if not sql_results:
        return "No SQL results available."

    if sql_results.get("error"):
        return f"SQL Error: {sql_results['error']}"

    columns = sql_results.get("columns", [])
    rows = sql_results.get("rows", [])
    row_count = sql_results.get("row_count", 0)

    if row_count == 0:
        return "SQL query returned 0 rows."

    # Build a readable table-like format
    lines = []
    lines.append(f"SQL Results ({row_count} rows):\n")
    lines.append(" | ".join(columns))
    lines.append("-" * (len(" | ".join(columns))))

    for row in rows[:50]:  # Cap at 50 rows for context window
        values = [str(row.get(col, "")) for col in columns]
        lines.append(" | ".join(values))

    if row_count > 50:
        lines.append(f"... and {row_count - 50} more rows")

    return "\n".join(lines)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  NEW: Query Router Node                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def query_router_node(state: RAGState) -> dict:
    """
    Classify the user query into one of three routes:
      - "sql"  → structured database query path
      - "rag"  → document retrieval path
      - "both" → both paths, merged at synthesis
    """
    query = state["query"]

    response = llm.invoke([
        SystemMessage(content=QUERY_ROUTER_PROMPT),
        HumanMessage(content=query),
    ])

    result_text = _extract_response_text(response.content)
    result_text = _strip_json_fences(result_text)

    route = "rag"  # safe default
    reason = ""
    try:
        parsed = json.loads(result_text)
        route = parsed.get("route", "rag").lower()
        reason = parsed.get("reason", "")
    except (json.JSONDecodeError, TypeError):
        logger.warning(
            "query_router_node: failed to parse, defaulting to 'rag'. "
            "Raw: %s", result_text[:200],
        )

    # Validate route value
    if route not in ("sql", "rag", "both"):
        logger.warning("query_router_node: invalid route '%s', defaulting to 'rag'", route)
        route = "rag"

    logger.info(
        "query_router_node: query='%s' → route=%s (%s)",
        query[:80], route, reason,
    )
    return {"query_route": route}


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  NEW: SQL Agent Node                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def sql_agent_node(state: RAGState) -> dict:
    """
    LLM call that decides which SQL tool to invoke.
    On first call, always calls sql_schema_inspector.
    On subsequent calls (after schema is in state), calls sql_query_executor.
    """
    query = state["query"]
    schema = state.get("sql_schema", "")
    messages = state["messages"]

    # Build the conversation for the SQL LLM
    system = SystemMessage(content=SQL_AGENT_SYSTEM_PROMPT)
    user = HumanMessage(content=query)

    if schema:
        # Schema already fetched — this is the second turn (execute query)
        # Include schema context and previous messages so the LLM can write the query
        context = HumanMessage(
            content=(
                f"Schema has already been retrieved. Use it to write your SQL query now.\n\n"
                f"Schema:\n{schema}\n\n"
                f"Write a SELECT query to answer: {query}"
            )
        )
        response = llm_with_sql_tools.invoke([system, context])
    else:
        # First turn — fetch schema
        response = llm_with_sql_tools.invoke([system, user])

    logger.info(
        "sql_agent_node: tool_calls=%s",
        response.tool_calls,
    )
    return {"messages": [response]}


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  NEW: SQL Tool Executor Node                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
sql_tool_node = ToolNode(tools=SQL_TOOLS)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  NEW: SQL Extract & Route Node                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def sql_extract_node(state: RAGState) -> dict:
    """
    After SQL tools execute, extract:
      - Schema info (from sql_schema_inspector) → cache in state
      - Query results (from sql_query_executor) → store in state
    Then decide: do we need another tool call, or are we done?
    """
    messages = state["messages"]

    # Check latest ToolMessage for schema
    schema = state.get("sql_schema", "")
    sql_results = state.get("sql_results", {})

    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            if msg.name == "sql_schema_inspector" and not schema:
                try:
                    parsed = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                    if isinstance(parsed, dict) and "tables" in parsed:
                        schema = json.dumps(parsed, indent=2)
                        logger.info("sql_extract_node: schema cached (%d tables)", len(parsed.get("tables", {})))
                except (json.JSONDecodeError, TypeError):
                    schema = msg.content

            elif msg.name == "sql_query_executor":
                try:
                    parsed = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                    if isinstance(parsed, dict):
                        sql_results = parsed
                        row_count = parsed.get("row_count", 0)
                        logger.info("sql_extract_node: got %d rows", row_count)
                except (json.JSONDecodeError, TypeError):
                    sql_results = {"raw": msg.content}

    updates = {}

    # Cache schema if we just got it
    if schema and not state.get("sql_schema"):
        updates["sql_schema"] = schema

    # If we have results, we're done — route to synthesize
    if sql_results.get("rows") is not None or sql_results.get("error"):
        updates["sql_results"] = sql_results
        return {**updates, "messages": []}  # empty messages = no more tool calls

    # If we got schema but no results yet, we need the agent to write a query
    if schema and not sql_results.get("rows"):
        # Signal that we need another agent turn by returning messages
        # The conditional edge will route back to sql_agent
        return updates

    return updates


def _sql_needs_more(state: RAGState) -> str:
    """
    After sql_extract, decide:
      - "sql_agent"  → got schema, need to write & execute query
      - "synthesizer" → got results, go to synthesis
    """
    sql_results = state.get("sql_results", {})
    schema = state.get("sql_schema", "")

    # If we have query results (or an error), we're done
    if sql_results.get("rows") is not None or sql_results.get("error"):
        return "synthesizer"

    # If we have schema but no results, agent needs to write the query
    if schema:
        return "sql_agent"

    # Fallback: go back to agent (shouldn't normally happen)
    return "sql_agent"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  NEW: SQL Direct Synthesize Node                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def sql_synthesize_node(state: RAGState) -> dict:
    """
    Synthesize a final answer using ONLY SQL results.
    Used when query_route is "sql" — skips rerank/score entirely.
    """
    query = state["original_query"]  # Use original query, not reformulated
    sql_results = state.get("sql_results", {})

    if sql_results.get("error"):
        answer = json.dumps({
            "answer": (
                f"I encountered an error querying the database: "
                f"{sql_results['error']}. Please try rephrasing your question "
                f"or contact support."
            ),
            "metadata": [],
        })
        return {"messages": [AIMessage(content=answer)], "synthesized_answer": answer}

    if not sql_results.get("rows"):
        answer = json.dumps({
            "answer": (
                "The database query returned no results. This could mean "
                "the data doesn't exist yet, or the query needs to be adjusted. "
                "Try asking about specific products, orders, or categories."
            ),
            "metadata": [],
        })
        return {"messages": [AIMessage(content=answer)], "synthesized_answer": answer}

    sql_context = _format_sql_results_as_context(sql_results)

    response = llm.invoke([
        SystemMessage(content=SQL_SYNTHESIZE_SYSTEM_PROMPT),
        HumanMessage(
            content=f"User question:\n{query}\n\n{sql_context}"
        ),
    ])

    answer_text = _extract_response_text(response.content)
    answer_text = _strip_json_fences(answer_text)

    logger.info("sql_synthesize_node: answer generated (%d chars)", len(answer_text))
    return {
        "messages": [AIMessage(content=answer_text)],
        "synthesized_answer": answer_text,
    }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  NEW: Combined Synthesize Node (RAG + SQL)                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def combined_synthesize_node(state: RAGState) -> dict:
    """
    Synthesize a final answer using BOTH RAG chunks and SQL results.
    Used when query_route is "both".
    """
    query = state["original_query"]
    chunks = state.get("reranked_chunks", [])
    sql_results = state.get("sql_results", {})

    # Build context from both sources
    context_parts = []

    if sql_results and sql_results.get("rows"):
        context_parts.append(
            "=== DATABASE RESULTS ===\n"
            + _format_sql_results_as_context(sql_results)
        )
    elif sql_results.get("error"):
        context_parts.append(
            f"=== DATABASE ERROR ===\n{sql_results['error']}"
        )

    if chunks:
        context_parts.append(
            "=== DOCUMENT CHUNKS ===\n"
            + _format_chunks_as_context(chunks)
        )

    if not context_parts:
        answer = json.dumps({
            "answer": "No data was available from either the database or documents to answer your question.",
            "metadata": [],
        })
        return {"messages": [AIMessage(content=answer)], "synthesized_answer": answer}

    full_context = "\n\n".join(context_parts)

    response = llm.invoke([
        SystemMessage(content=COMBINED_SYNTHESIZE_SYSTEM_PROMPT),
        HumanMessage(
            content=f"User question:\n{query}\n\n{full_context}"
        ),
    ])

    answer_text = _extract_response_text(response.content)
    answer_text = _strip_json_fences(answer_text)

    logger.info("combined_synthesize_node: answer generated (%d chars)", len(answer_text))
    return {
        "messages": [AIMessage(content=answer_text)],
        "synthesized_answer": answer_text,
    }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  EXISTING Graph Nodes (unchanged logic, minor naming updates)            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def agent_node(state: RAGState) -> dict:
    """
    First LLM call — decides which retrieval tool to invoke.
    On retry, start a FRESH conversation (no old AIMessage/ToolMessage pairs).
    """
    messages = state["messages"]
    query = state["query"]
    retries = state.get("retries", 0)

    system_prompt = AGENT_RETRY_SYSTEM_PROMPT if retries > 0 else AGENT_SYSTEM_PROMPT
    system = SystemMessage(content=system_prompt)
    user = HumanMessage(content=query)

    if retries > 0:
        response = llm_with_retrieval_tools.invoke([system, user])
    else:
        response = llm_with_retrieval_tools.invoke([system] + messages + [user])

    logger.info("agent_node (retry=%d): tool_calls=%s", retries, response.tool_calls)
    return {"messages": [response]}


def rerank_node(state: RAGState) -> dict:
    raw_chunks = state["raw_chunks"]
    query = state["query"]

    if not raw_chunks:
        return {"reranked_chunks": []}

    if ENABLE_RERANK:
        try:
            documents = [_extract_text_from_chunk(c) for c in raw_chunks]
            reranked = rerank_chunks(
                query=query,
                documents=documents,
                original_chunks=raw_chunks,
                top_n=RERANK_TOP_N,
                model=RERANK_MODEL,
            )
            logger.info(
                "rerank_node: %d → %d chunks (model=%s)",
                len(raw_chunks), len(reranked), RERANK_MODEL,
            )
            return {"reranked_chunks": reranked}
        except Exception as exc:
            logger.warning("rerank_node failed, using original order: %s", exc)

    return {"reranked_chunks": raw_chunks}


def score_node(state: RAGState) -> dict:
    chunks = state["reranked_chunks"]
    query = state["query"]

    if not chunks:
        return {"metadata": []}

    chunk_texts = [_extract_text_from_chunk(c) for c in chunks]
    computed_scores = score_chunks(query, chunk_texts)

    metadata_list = [
        _build_metadata_entry(chunk, rank=i, scores=scores)
        for i, (chunk, scores) in enumerate(zip(chunks, computed_scores), start=1)
    ]

    logger.info(
        "score_node: scored %d chunks (best relevance=%.4f)",
        len(metadata_list),
        metadata_list[0]["relevance_score"] if metadata_list else 0.0,
    )
    return {"metadata": metadata_list}


def synthesize_node(state: RAGState) -> dict:
    """
    RAG-only synthesis. Generates final answer using reranked & scored chunks.
    Stores the raw answer in state for the relevance evaluator.
    """
    chunks = state["reranked_chunks"]
    query = state["original_query"]  # Use original query

    if not chunks:
        answer = json.dumps({
            "answer": (
                "I could not retrieve any relevant documents to answer your "
                "question, even after reformulating the query. Please try "
                "rephrasing with more specific financial terms or "
                "contact support for further assistance."
            ),
            "metadata": [],
        })
        fallback = AIMessage(content=answer)
        return {
            "messages": [fallback],
            "synthesized_answer": answer,
        }

    context = _format_chunks_as_context(chunks)

    system = SystemMessage(content=SYNTHESIZE_SYSTEM_PROMPT)
    user = HumanMessage(
        content=f"User question:\n{query}\n\nDocument chunks:\n{context}"
    )

    response = llm.invoke([system, user])
    answer_text = _extract_response_text(response.content)
    answer_text = _strip_json_fences(answer_text)

    logger.info("synthesize_node: answer generated (%d chars)", len(answer_text))

    clean_message = AIMessage(content=answer_text)
    return {
        "messages": [clean_message],
        "synthesized_answer": answer_text,
    }


def answer_relevance_node(state: RAGState) -> dict:
    """
    Check whether the retrieved data (chunks and/or SQL results) are relevant
    to the user's query. For SQL-only queries, skip this check (data is always
    "relevant" if rows were returned — the question is whether the query was
    right, which the router already handled).
    """
    query = state["original_query"]
    route = state.get("query_route", "rag")

    # SQL-only queries: skip relevance check — trust the SQL results
    if route == "sql":
        logger.info("answer_relevance_node: skipping for SQL route")
        return {"answer_relevant": True}

    # For RAG and "both" routes, check chunk relevance
    chunks = state.get("reranked_chunks", [])

    if not chunks:
        return {"answer_relevant": False}

    chunk_lines = []
    for i, chunk in enumerate(chunks, start=1):
        text = _extract_text_from_chunk(chunk)
        meta = _get_chunk_metadata(chunk)
        section = meta.get("section_header", "")
        doc = meta.get("document_name", "")
        page = meta.get("page_number", "?")
        modality = meta.get("modality", "text")

        preview = text[:300] + "..." if len(text) > 300 else text
        chunk_lines.append(
            f"Chunk {i} [{modality}, {doc} p.{page}, Section: {section}]:\n{preview}"
        )

    chunks_text = "\n\n".join(chunk_lines)

    prompt = CHUNK_RELEVANCE_PROMPT.format(
        query=query,
        chunks_text=chunks_text,
    )

    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Evaluate chunk relevance."),
    ])

    result_text = _extract_response_text(response.content)
    result_text = _strip_json_fences(result_text)

    relevant = False
    try:
        parsed = json.loads(result_text)
        relevant = bool(parsed.get("relevant", False))
    except (json.JSONDecodeError, TypeError):
        logger.warning("answer_relevance_node: failed to parse: %s", result_text[:200])
        relevant = True

    logger.info(
        "answer_relevance_node: query='%s' | %d chunks | relevant=%s",
        query[:60], len(chunks), relevant,
    )
    return {"answer_relevant": relevant}


def query_reformulate_node(state: RAGState) -> dict:
    """
    Rewrite the query for better retrieval. Only applies to the RAG path —
    SQL queries don't need reformulation (the SQL agent handles that).
    """
    query = state["original_query"]  # Always reformulate from the ORIGINAL
    retries = state.get("retries", 0)
    new_retries = retries + 1

    chunks = state.get("reranked_chunks", [])

    reformulate_context = query
    if chunks:
        chunk_summary_parts = []
        for chunk in chunks[:3]:
            text = _extract_text_from_chunk(chunk)
            meta = _get_chunk_metadata(chunk)
            section = meta.get("section_header", "")
            chunk_summary_parts.append(f"Section '{section}': {text[:150]}")
        chunk_summary = " | ".join(chunk_summary_parts)

        reformulate_context = (
            f"Original query: {query}\n\n"
            f"Previous retrieval returned chunks about the WRONG topic: {chunk_summary}\n\n"
            f"Reformulate the query to find documents about the correct topic."
        )
    else:
        reformulate_context = (
            f"Original query: {query}\n\n"
            f"Previous retrieval returned no results.\n"
            f"Reformulate the query."
        )

    response = llm.invoke([
        SystemMessage(content=QUERY_REFORMULATE_SYSTEM_PROMPT),
        HumanMessage(content=reformulate_context),
    ])

    reformulated = _extract_response_text(response.content).strip().strip('"').strip("'")
    logger.info(
        "query_reformulate_node: attempt %d/%d — original: '%s' → reformulated: '%s'",
        new_retries, MAX_RETRIES, query, reformulated,
    )

    context_msg = HumanMessage(
        content=(
            f"[Query Reformulation — attempt {new_retries} of {MAX_RETRIES}] "
            f"Previous attempt returned {'wrong-topic chunks' if chunks else 'no chunks'}. "
            f"Please use this reformulated query for retrieval:\n"
            f"\"{reformulated}\""
        )
    )

    return {
        "query": reformulated,
        "retries": new_retries,
        "messages": [context_msg],
        "synthesized_answer": "",
        "answer_relevant": False,
    }


def extract_chunks_node(state: RAGState) -> dict:
    """Extract raw chunks from the latest ToolMessage."""
    chunks = _extract_chunks_from_messages(state["messages"])
    logger.info("extract_chunks: found %d raw chunks", len(chunks))
    return {"raw_chunks": chunks}


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Conditional Edge Functions                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def _route_by_query_type(state: RAGState) -> str:
    """
    After query_router_node, dispatch to the correct path.

    Returns:
        "sql"   → sql_agent (SQL-only path)
        "rag"   → agent (RAG path)
        "both"  → both_parallel (both paths, merged later)
    """
    route = state.get("query_route", "rag")
    return route


def _has_relevant_chunks(state: RAGState) -> str:
    chunks = state.get("raw_chunks", [])
    retries = state.get("retries", 0)

    if chunks:
        return "rerank"
    if retries < MAX_RETRIES:
        logger.info("No chunks (retries=%d/%d) → reformulate", retries, MAX_RETRIES)
        return "reformulate"

    logger.warning("No chunks, retries exhausted (%d/%d) → fallback synthesize", retries, MAX_RETRIES)
    return "synthesize"


def _check_answer_relevance(state: RAGState) -> str:
    relevant = state.get("answer_relevant", False)
    retries = state.get("retries", 0)
    route = state.get("query_route", "rag")

    # SQL-only: always end (relevance was skipped)
    if route == "sql":
        return "end"

    if relevant:
        logger.info("Answer is relevant → END")
        return "end"

    if retries < MAX_RETRIES:
        logger.info(
            "Answer NOT relevant (retries=%d/%d) → reformulate",
            retries, MAX_RETRIES,
        )
        return "reformulate"

    logger.warning(
        "Answer NOT relevant, retries exhausted (%d/%d) → END with current answer",
        retries, MAX_RETRIES,
    )
    return "end"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  NEW: Both-Path Merge Node                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def both_merge_node(state: RAGState) -> dict:
    """
    After both the RAG path and SQL path have completed, this node
    checks if we have data from both sources. If RAG chunks are empty
    (retrieval failed), we still proceed with SQL results alone.

    This is a pass-through node that lets the combined_synthesize_node
    handle the merge logic.
    """
    sql_results = state.get("sql_results", {})
    chunks = state.get("reranked_chunks", [])

    logger.info(
        "both_merge_node: sql_rows=%s, rag_chunks=%d",
        sql_results.get("row_count", 0) if sql_results else 0,
        len(chunks),
    )

    # If we have both, great. If only one, still proceed.
    # The combined_synthesize_node handles missing data gracefully.
    return {}


def _both_has_chunks(state: RAGState) -> str:
    """
    After the RAG path completes in 'both' mode, decide next step.
    Always goes to both_merge since SQL path is independent.
    """
    return "both_merge"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Graph Construction                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph RAG + SQL pipeline.

    Pipeline:
        query_router
            ├── "sql"  → sql_agent → sql_tools → sql_extract
            │               ├── [need query] → sql_agent (loop)
            │               └── [has results] → sql_synthesize → evaluate_relevance → END
            │
            ├── "rag"  → agent → tools → extract_chunks
            │               ├── [chunks] → rerank → score → synthesize → evaluate_relevance
            │               │       ├── [relevant] → END
            │               │       └── [not relevant + retries] → reformulate → agent
            │               └── [no chunks + retries] → reformulate → agent
            │               └── [no chunks + exhausted] → synthesize → END
            │
            └── "both" → Fan-out:
                    RAG:  agent → tools → extract_chunks → [chunks?] → rerank → score → both_merge
                    SQL:  sql_agent → sql_tools → sql_extract → sql_synthesize_ready → both_merge
                    → combined_synthesize → evaluate_relevance → END
    """
    graph = StateGraph(RAGState)

    # ─── Nodes ─────────────────────────────────────────────────────────────
    # Router
    graph.add_node("query_router", query_router_node)

    # RAG path
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools=RETRIEVAL_TOOLS))
    graph.add_node("extract_chunks", extract_chunks_node)
    graph.add_node("query_reformulate", query_reformulate_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("score", score_node)
    graph.add_node("synthesize", synthesize_node)

    # SQL path
    graph.add_node("sql_agent", sql_agent_node)
    graph.add_node("sql_tools", sql_tool_node)
    graph.add_node("sql_extract", sql_extract_node)
    graph.add_node("sql_synthesize", sql_synthesize_node)

    # Both path
    graph.add_node("both_merge", both_merge_node)
    graph.add_node("combined_synthesize", combined_synthesize_node)

    # Shared
    graph.add_node("evaluate_relevance", answer_relevance_node)

    # ─── Edges ─────────────────────────────────────────────────────────────
    graph.add_edge(START, "query_router")

    # ── Router dispatches to three paths ──
    graph.add_conditional_edges(
        "query_router",
        _route_by_query_type,
        {
            "sql": "sql_agent",
            "rag": "agent",
            "both": "agent",  # Start RAG path first; SQL runs in parallel via "both" handling
        },
    )

    # ── SQL path ──
    graph.add_conditional_edges(
        "sql_agent",
        tools_condition,
        {"tools": "sql_tools", "__end__": END},
    )
    graph.add_edge("sql_tools", "sql_extract")
    graph.add_conditional_edges(
        "sql_extract",
        _sql_needs_more,
        {
            "sql_agent": "sql_agent",       # Got schema, need to write query
            "synthesizer": "sql_synthesize",  # Got results, synthesize
        },
    )

    # ── RAG path (same as before) ──
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", "__end__": END},
    )
    graph.add_edge("tools", "extract_chunks")
    graph.add_conditional_edges(
        "extract_chunks",
        _has_relevant_chunks,
        {
            "rerank": "rerank",
            "reformulate": "query_reformulate",
            "synthesize": "synthesize",
        },
    )
    graph.add_edge("query_reformulate", "agent")
    graph.add_edge("rerank", "score")

    # ── RAG path: after scoring, decide synthesize target ──
    graph.add_conditional_edges(
        "score",
        lambda state: "combined_synthesize" if state.get("query_route") == "both" else "synthesize",
        {
            "synthesize": "synthesize",
            "combined_synthesize": "combined_synthesize",
        },
    )

    # ── For "both" route: combined_synthesize also needs SQL results ──
    # When route is "both", we run SQL path AFTER RAG scoring completes.
    # combined_synthesize waits for sql_results in state.
    # To handle this, we add a conditional edge that checks if SQL is done:
    graph.add_conditional_edges(
        "combined_synthesize",
        lambda state: "evaluate_relevance" if state.get("sql_results") else "sql_agent",
        {
            "sql_agent": "sql_agent",           # Need to run SQL first
            "evaluate_relevance": "evaluate_relevance",  # Both sources ready
        },
    )

    # After SQL path completes in "both" mode, route back to combined synthesize
    graph.add_conditional_edges(
        "sql_synthesize",
        lambda state: "combined_synthesize" if state.get("query_route") == "both" else "evaluate_relevance",
        {
            "combined_synthesize": "combined_synthesize",
            "evaluate_relevance": "evaluate_relevance",
        },
    )

    # ── Synthesis → Relevance check (shared) ──
    graph.add_edge("synthesize", "evaluate_relevance")
    graph.add_conditional_edges(
        "evaluate_relevance",
        _check_answer_relevance,
        {
            "end": END,
            "reformulate": "query_reformulate",
        },
    )

    # Fallback: synthesize with no chunks → END directly
    graph.add_edge("synthesize", END)

    return graph.compile()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Compiled Graph Singleton                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
rag_graph = build_graph()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Main Entry Point                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def run_rag_agent(query: str, k: int = 10) -> QueryResponse:
    """
    Stateless RAG + SQL agent entry point.

    Args:
        query: User's natural language question.
        k: Number of top-k chunks to retrieve (passed to RAG tools).

    Returns:
        QueryResponse with:
          - answer: LLM-synthesized from reranked chunks and/or SQL results
          - metadata: programmatic scores + ingestion metadata + provenance
    """
    initial_state: RAGState = {
        "messages": [],
        "query": query,
        "original_query": query,     # Preserve original for synthesis
        "query_route": "rag",        # Will be set by query_router
        "raw_chunks": [],
        "reranked_chunks": [],
        "metadata": [],
        "retries": 0,
        "synthesized_answer": "",
        "answer_relevant": False,
        # SQL fields
        "sql_results": {},
        "sql_schema": "",
    }

    final_state = rag_graph.invoke(initial_state)
    messages = final_state.get("messages", [])
    metadata = final_state.get("metadata", [])

    # Extract synthesized answer from last AI message
    answer = "I could not generate a proper answer."
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            content = _extract_response_text(msg.content).strip()
            if not content:
                continue

            content = _strip_json_fences(content)

            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "answer" in parsed:
                    answer = parsed["answer"]
                    break
            except (json.JSONDecodeError, TypeError):
                answer = content
                break

    return QueryResponse(
        query=query,
        answer=answer,
        metadata=metadata,
    )
