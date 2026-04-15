"""
Integrated RAG + SQL Agent with 3-way routing.

Pipeline overview:
    user query
        |
    ROUTER (LLM classifies)
       / | \
      /  |  \
   SQL  DOCUMENT  HYBRID
     |     |        |
     |     |     sql_write -> sql_execute -> agent -> tools -> extract
     |     |       -> rerank -> score -> hybrid_merge -> hybrid_synthesize
     |     |
     |   agent -> tools -> extract
     |     -> rerank -> score -> synthesize -> evaluate_relevance -> retry?
     |
   sql_write -> sql_execute -> sql_synthesize

Response shape (QueryResponse — UNCHANGED):
    query: str
    answer: str
    metadata: List[MetadataEntry]   <-- SQL data lives here as extra fields

SQL metadata entry (leverages extra="allow" on MetadataEntry):
    rank=1, modality="text", content=sql_answer,
    route="sql", sql_query="...", sql_row_count=N

Consumers check entry.route to distinguish SQL from document entries.
"""

import json
import logging
import os
from decimal import Decimal
from typing import Annotated, Any, Dict, List, Optional, TypedDict, Union

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

import psycopg2
from psycopg2.extras import RealDictCursor

from api.v1.schemas.query_schema import QueryResponse, MetadataEntry
from api.v1.tools.fts_search_tool import fts_search
from api.v1.tools.hybrid_search_tool import hybrid_search
from api.v1.tools.vector_search_tool import vector_search
from api.v1.utils.reranker import rerank_chunks
from api.v1.utils.scoring import score_chunks
from core.helper import get_db_schema

load_dotenv(override=True)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ENABLE_RERANK: bool = os.getenv("ENABLE_RERANK", "true").lower() == "true"
RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", "10"))
RERANK_MODEL: str = os.getenv("RERANK_MODEL", "rerank-v3.5")
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
MODEL_NAME = os.getenv("GEMINI_MODEL")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is required")

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)

# ---------------------------------------------------------------------------
# Database schema (cached at import time for the SQL path)
# ---------------------------------------------------------------------------
try:
    _DB_SCHEMA = get_db_schema()
    logger.info("SQL agent schema loaded (%d chars)", len(_DB_SCHEMA))
except Exception as exc:
    logger.warning("Failed to load DB schema at startup: %s", exc)
    _DB_SCHEMA = ""

DATABASE_URL = os.getenv("ADMIN_DB_URL", "")

# ---------------------------------------------------------------------------
# Retrieval tools — bound for the document path's agent node
# ---------------------------------------------------------------------------
RETRIEVAL_TOOLS: List[BaseTool] = [vector_search, fts_search, hybrid_search]
llm_with_tools = llm.bind_tools(RETRIEVAL_TOOLS)





# ── ADD THESE IMPORTS ──────────────────────────────────────────────────
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from langgraph.checkpoint.memory import MemorySaver  # ← Checkpointer
# ───────────────────────────────────────────────────────────────────────

# ── ADD LOGGING CONFIGURATION (after load_dotenv) ──────────────────────
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Rotating file handler: 10MB max, keep 3 backup files
file_handler = RotatingFileHandler(
    LOG_DIR / "rag_agent.log",
    maxBytes=10*1024*1024,  # 10 MB
    backupCount=3,
    encoding="utf-8"
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
)

# Console handler for dev
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Silence noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
# ───────────────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# Prompts — Router
# ---------------------------------------------------------------------------
ROUTER_SYSTEM_PROMPT = """You are a query router for a RELIANCE INDUSTRIES financial & user analytics assistant.

You decide whether a user question should be answered using:
1. **SQL** — structured data from database tables (user metrics, shareholdings, 
   KPIs, comparisons, averages, percentages, rankings, trends, real-time analytics).
2. **DOCUMENT** — unstructured content from PDFs (narrative analysis, management 
   commentary, qualitative descriptions, policies, reports).
3. **HYBRID** — questions needing BOTH structured metrics AND narrative context.

## Available SQL Tables

### Financial Tables:
- jio_users, reliance_shareholders

### User & Shareholder Analytics Tables (REAL-TIME DATA):

#### TABLE: jio_users (Telecom Subscriber Analytics)
Columns:
  - user_id (UUID), mobile_number (VARCHAR), email (VARCHAR)
  - subscription_plan: 'JioPrime', 'JioPostpaid Plus', 'JioPrepaid', 'JioFiber', 'JioAirFiber'
  - plan_category: 'Prepaid', 'Postpaid', 'Fiber', 'Enterprise'
  - region: 'North', 'South', 'East', 'West', 'Central', 'Metro', 'Tier-2', 'Tier-3'
  - circle: 'Mumbai', 'Delhi', 'Kolkata', 'Chennai', 'Bangalore', 'Hyderabad', 'Pune', 'Ahmedabad'
  - data_usage_gb_30d (DECIMAL): Data consumed in last 30 days
  - voice_minutes_30d (INT), sms_count_30d (INT)
  - arpu_inr (DECIMAL): Average Revenue Per User in INR
  - recharge_frequency_days (INT): Avg days between recharges
  - churn_risk_score (DECIMAL 0-1): ML model prediction (higher = more likely to churn)
  - nps_score (INT): -100 to 100 customer satisfaction
  - linked_jfs_account (BOOLEAN): Has Jio Financial Services account
  - jfs_shares_held (INT): JFS equity shares owned by this user
  - last_activity_timestamp (TIMESTAMPTZ): Last seen online
  - is_active (BOOLEAN): Currently active subscriber
  - registration_date (DATE)

#### TABLE: reliance_shareholders (Shareholder Registry)
Columns:
  - user_id (UUID), holder_name (VARCHAR)
  - holder_type: 'Retail Investor', 'Institutional', 'Promoter', 'FII/DII', 'Employee ESOP'
  - pan_number (VARCHAR), demat_account (VARCHAR)
  - shares_held (BIGINT): Number of RIL equity shares
  - share_category: 'Equity', 'Preference', 'Convertible Debenture', 'Warrants', 'ESOP'
  - avg_buy_price (DECIMAL): Average purchase price per share (INR)
  - current_value_in_cr (DECIMAL): Current holding value in INR Crore
  - percentage_holding (DECIMAL): % of total outstanding RIL shares
  - voting_rights (BIGINT): Voting power (1 share = 1 vote for Equity)
  - region (VARCHAR): Geographic region of shareholder
  - is_active (BOOLEAN): Account active status
  - kyc_status: 'Verified', 'Pending', 'Expired'
  - first_purchase_date (DATE), last_transaction_date (DATE)

## Decision Rules — Route to **SQL** for:

### 📱 Jio User Queries (Real-time Analytics):
* "How many active Jio users in Maharashtra?" → COUNT + region filter
* "What is the average ARPU for JioPostpaid Plus subscribers?" → AVG(arpu_inr) + plan filter
* "Show me Jio users with high churn risk (>0.8) in Tier-2 cities" → WHERE + ranking
* "List top 10 circles by data usage in last 30 days" → GROUP BY + ORDER BY + LIMIT
* "How many users recharged in the last 7 days?" → Date filter on last_activity
* "What percentage of Jio users have linked JFS accounts?" → Aggregation + percentage
* "Compare ARPU between Prepaid and Postpaid users by region" → GROUP BY + comparison
* "Users who haven't been active in last 30 days" → Timestamp filter
* "Distribution of NPS scores across subscription plans" → Statistical aggregation

### 📊 Shareholder Queries (Holdings & Voting):
* "What percentage does Promoter group hold in RIL?" → SUM(percentage_holding) WHERE holder_type='Promoter'
* "Show institutional shareholders with >1% holding" → WHERE + filter
* "Total voting rights held by FII/DII category" → SUM(voting_rights) + filter
* "Shareholders who bought shares in last 30 days" → Date filter on last_transaction_date
* "Top 10 shareholders by current value in Crore" → ORDER BY + LIMIT
* "Average buy price vs current market value for Retail investors" → Aggregation + comparison
* "How many shareholders have pending KYC?" → COUNT + kyc_status filter
* "Regional distribution of shareholder holdings" → GROUP BY region + SUM

### 🔁 Real-time / Operational KPIs:
* "Active users by region right now" → WHERE is_active=true + last_activity within threshold
* "Jio users with jfs_shares_held > 100 AND high data usage" → Multi-condition filter
* "Shareholders with KYC expired in last quarter" → Date range filter
* "Churn risk distribution by subscription plan" → GROUP BY + statistical summary

### 📈 Comparisons, Trends & Aggregations:
* YoY/QoQ user growth: Compare registration_date periods
* ARPU trends by plan: AVG(arpu_inr) GROUP BY subscription_plan
* Holding changes: Compare first_purchase_date vs last_transaction_date metrics
* Rankings: Top regions, top plans, top shareholders, highest churn risk
* Filters: active/inactive, verified KYC, specific date ranges, value thresholds

## Route to **DOCUMENT** for:
* "Explain Jio's subscriber acquisition strategy" → Narrative content
* "What does management say about shareholder value creation?" → Commentary
* "Describe the Jio Financial Services integration with telecom" → Policy/strategy docs
* "Risk factors related to user churn or regulatory changes" → Risk disclosures
* "Qualitative analysis of market share trends" → Management discussion
* "ESG policies affecting shareholder relations" → Governance documents

## Route to **HYBRID** for:
* "What is the promoter holding AND why did it change recently?" → SQL number + document explanation
* "Show ARPU trends for JioPostpaid AND management commentary on pricing strategy" → Metric + context
* "How many BP partners in Gujarat AND what is the expansion strategy there?" → Count + narrative
* "List top 5 shareholders by holding AND any recent disclosures about their stakes" → Data + news
* "What is the churn risk score distribution AND what initiatives are planned to reduce churn?" → Analytics + strategy
* Any query combining precise user/shareholder metrics with narrative explanation or forward-looking guidance

## Output Format
Return ONLY a JSON object with exactly one key:
  {"route": "sql"} | {"route": "document"} | {"route": "hybrid"}

No markdown fences. No explanations. Only the JSON object."""

# ---------------------------------------------------------------------------
# Prompts — SQL Agent
# ---------------------------------------------------------------------------
SQL_QUERY_WRITER_PROMPT = """You are a SQL expert for RELIANCE INDUSTRIES financial AND user analytics data.

## Database Schema

### Financial Tables:
{schema}

### User & Shareholder Analytics Tables:

-- TABLE: jio_users (Telecom Subscriber Analytics)
CREATE TABLE jio_users (
    user_id UUID PRIMARY KEY,
    mobile_number VARCHAR(20) UNIQUE,
    email VARCHAR(255),
    subscription_plan VARCHAR(100), -- 'JioPrime', 'JioPostpaid Plus', 'JioPrepaid', 'JioFiber', 'JioAirFiber'
    plan_category VARCHAR(50), -- 'Prepaid', 'Postpaid', 'Fiber', 'Enterprise'
    registration_date DATE,
    last_activity_timestamp TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE,
    region VARCHAR(50), -- 'North', 'South', 'East', 'West', 'Central', 'Metro', 'Tier-2', 'Tier-3'
    circle VARCHAR(100), -- 'Mumbai', 'Delhi', 'Kolkata', 'Chennai', 'Bangalore', etc.
    data_usage_gb_30d DECIMAL(10,2), -- Data consumed in last 30 days
    voice_minutes_30d INTEGER,
    sms_count_30d INTEGER,
    arpu_inr DECIMAL(10,2), -- Average Revenue Per User in INR
    recharge_frequency_days INTEGER,
    linked_jfs_account BOOLEAN DEFAULT FALSE,
    jfs_shares_held INTEGER DEFAULT 0, -- JFS equity shares owned
    loyalty_points INTEGER DEFAULT 0,
    churn_risk_score DECIMAL(3,3), -- ML model: 0.0 to 1.0 (higher = more likely to churn)
    nps_score INTEGER, -- -100 to 100
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- TABLE: reliance_shareholders (Shareholder Registry)
CREATE TABLE reliance_shareholders (
    user_id UUID PRIMARY KEY,
    holder_name VARCHAR(255) NOT NULL,
    holder_type VARCHAR(100), -- 'Retail Investor', 'Institutional', 'Promoter', 'FII/DII', 'Employee ESOP'
    pan_number VARCHAR(20),
    demat_account VARCHAR(100),
    shares_held BIGINT, -- Number of RIL equity shares
    share_category VARCHAR(100), -- 'Equity', 'Preference', 'Convertible Debenture', 'Warrants', 'ESOP'
    avg_buy_price DECIMAL(10,2), -- Average purchase price per share (INR)
    current_value_in_cr DECIMAL(15,4), -- Current holding value in INR Crore
    percentage_holding DECIMAL(8,4), -- % of total outstanding RIL shares
    voting_rights BIGINT, -- Voting power (1 share = 1 vote for Equity)
    region VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    kyc_status VARCHAR(20), -- 'Verified', 'Pending', 'Expired'
    first_purchase_date DATE,
    last_transaction_date DATE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

## Instructions
1. Write a SINGLE correct PostgreSQL SELECT query to answer the user's question.
2. Use ONLY the tables and columns shown in the schema above.
3. For jio_users:
   - Use `is_active = TRUE` to filter active subscribers
   - Use `last_activity_timestamp > NOW() - INTERVAL 'X days'` for "recently active"
   - ARPU is in `arpu_inr` column (INR), format as "₹{{value}}" in final answer
   - churn_risk_score is 0.0-1.0; >0.7 = high risk, >0.5 = medium, <0.3 = low
   - Use `linked_jfs_account = TRUE` to find users with JFS accounts
4. For reliance_shareholders:
   - Monetary values: `current_value_in_cr` is already in INR Crore
   - Use `percentage_holding` for ownership % queries
   - Use `voting_rights` for governance/voting power questions
   - Filter by `holder_type` for category-specific queries (Promoter, FII/DII, etc.)
   - Use `kyc_status` for compliance-related queries
5. Periods & Dates:
   - Use DATE/TIMESTAMPTZ columns with INTERVAL for time-based filters
   - Examples: `registration_date >= '2024-01-01'`, `last_activity_timestamp > NOW() - INTERVAL '7 days'`
6. Use ORDER BY for ranking questions (top-N, highest churn risk, largest holdings, etc.)
7. Use LIMIT 20 to keep results manageable (unless user explicitly asks for more)
8. For aggregations: Use COUNT, SUM, AVG, ROUND appropriately
9. Return ONLY the SQL query. No explanations, no markdown fences, no backticks.

## Examples

User: "How many active Jio users in Maharashtra?"
→ SELECT COUNT(*) FROM jio_users WHERE region = 'West' AND is_active = TRUE;

User: "What is the average ARPU for JioPostpaid Plus subscribers?"
→ SELECT ROUND(AVG(arpu_inr), 2) as avg_arpu FROM jio_users WHERE subscription_plan = 'JioPostpaid Plus' AND is_active = TRUE;

User: "Show top 5 shareholders by percentage holding"
→ SELECT holder_name, holder_type, percentage_holding, current_value_in_cr FROM reliance_shareholders ORDER BY percentage_holding DESC LIMIT 5;

User: "How many shareholders have pending KYC?"
→ SELECT COUNT(*) FROM reliance_shareholders WHERE kyc_status = 'Pending';

User: "Jio users with high churn risk (>0.8) in Tier-2 regions"
→ SELECT circle, COUNT(*) as user_count, ROUND(AVG(arpu_inr), 2) as avg_arpu FROM jio_users WHERE churn_risk_score > 0.8 AND region IN ('Tier-2', 'Tier-3') AND is_active = TRUE GROUP BY circle ORDER BY user_count DESC LIMIT 10;

User: "Total voting rights held by Institutional shareholders"
→ SELECT SUM(voting_rights) as total_voting_rights, COUNT(*) as shareholder_count FROM reliance_shareholders WHERE holder_type = 'Institutional' AND is_active = TRUE;

## User Question
{question}

## SQL Query"""


SQL_ANSWER_SYNTHESIZER_PROMPT = """You are a RELIANCE INDUSTRIES financial AND user analytics assistant.

You will receive:
- A user's question
- The SQL query that was executed
- The raw query results

Provide a clear, professional answer. Rules:

### 📊 Formatting Rules:
1. **Monetary Values**:
   - For financial tables: Use INR Crore notation (e.g., "₹258,027 Crore")
   - For jio_users ARPU: Use "₹{{value}}" format (e.g., "₹342.50")
   - For shareholder current_value_in_cr: Already in Crore, format as "₹{{value}} Crore"

2. **Percentages & Ratios**:
   - Always show % with one decimal: "23.4%"
   - For churn_risk_score: Convert to percentage "0.85 → 85% churn risk"

3. **Time Periods**:
   - Always mention the timeframe: "in last 30 days", "as of {{date}}", "Q2 FY25"
   - For real-time queries: "currently active", "as of today"

4. **User Counts & Rankings**:
   - Use commas for large numbers: "12,450 users"
   - For top-N results: Use numbered list or mini-table

5. **Shareholder Data**:
   - Show holder_type context: "Promoter group holds...", "FII/DII category..."
   - Include voting rights context when relevant: "representing X% of total voting power"

6. **Risk & Compliance Fields**:
   - churn_risk_score: Interpret as Low (<30%), Medium (30-70%), High (>70%)
   - kyc_status: Mention compliance implications: "X shareholders with pending KYC require attention"

7. **If query returned no rows**: Say "No data found matching these criteria" and suggest alternatives.

8. **Add brief analytical context** when possible:
   - "ARPU of ₹342 for Postpaid is 2.3x higher than Prepaid (₹148), reflecting premium positioning"
   - "Promoter holding of 50.3% provides controlling interest with Y voting rights"

### 📋 Output Structure:
- Start with direct answer to the question
- Present numbers in bullet points or mini-table for multi-row results
- Add 1-2 sentences of analytical context if data supports it
- Mention data freshness: "Based on real-time subscriber data as of {{timestamp}}"

## User Question
{question}

## SQL Query Used
{sql_query}

## Query Results
{results}

## Answer"""
# ---------------------------------------------------------------------------
# Prompts — Document RAG Agent (unchanged from original)
# ---------------------------------------------------------------------------
AGENT_SYSTEM_PROMPT = """You are an expert Financial Document assistant.

You have access to three retrieval tools:
  - vector_search  -> best for natural language / conceptual questions
  - fts_search     -> best for codes, IDs, abbreviations, exact keywords  
  - hybrid_search  -> best for short or ambiguous queries

Rules:
1. Choose exactly ONE tool based on the query type.
2. Call the tool with the ORIGINAL user query (do not modify it).
3. Return ONLY a tool call - do not answer the question yourself.
4. The tool will return document chunks with metadata including:
   - modality: "text", "table", or "image"
   - section_header: the document section this chunk belongs to
   - For tables: table_data, docling_caption, llm_generated_caption
   - For images: llm_generated_caption describing the chart/figure
5. After retrieval, you will synthesize an answer using ONLY the returned chunks."""

AGENT_RETRY_SYSTEM_PROMPT = """You are an expert Financial Document assistant.

The previous retrieval attempt returned NO relevant document chunks.

You have access to three retrieval tools:
  - vector_search  -> best for natural language / conceptual questions
  - fts_search     -> best for codes, IDs, abbreviations, exact keywords
  - hybrid_search  -> best for short or ambiguous queries

Rules:
1. Choose exactly ONE tool and call it with the REFORMULATED query 
   provided in the most recent human message.
2. You MUST return a tool call - do not answer the question yourself.
3. The reformulated query is optimized for document retrieval."""

QUERY_REFORMULATE_SYSTEM_PROMPT = """You rewrite user questions for better document retrieval in a financial report system.

Your job:
1. Read the user's original question.
2. Express what the user is already asking - do NOT change the core meaning or intent.
3. Rewrite it as a natural sentence that sounds like something a financial report would discuss.
4. If the question has multiple aspects, split them into separate short keyword-style queries joined by " OR ".
5. Do NOT add a rigid structure or template. Just say it plainly in report language.

Examples:
  User: "how much was spent on marketing"
  -> "marketing expenditure OR advertising spend OR promotional costs"

  User: "what is the consumer base growth"
  -> "customer base growth OR subscriber growth OR user metrics trend"

  User: "tell me about profit"
  -> "profit OR EBITDA OR net income OR PAT"

  User: "what happened with revenue in Q2"
  -> "Q2 revenue OR second quarter top-line OR Q2 FY turnover"

Return ONLY the reformulated query. Nothing else."""

SYNTHESIZE_SYSTEM_PROMPT = """You are an expert financial document assistant.

You will be given document chunks and a user question. Provide the BEST
possible answer using the available chunks.

## Rules

1. DIRECT MATCH: If a chunk directly answers the question, use it.

2. PARTIAL / RELATED MATCH (CRITICAL): If no chunk has the exact answer
   BUT contains RELATED data, you MUST provide that related data and
   explain how it connects. Do NOT say "no information found" when
   related metrics exist.

3. FINANCIAL TERM MAPPING - treat these as equivalent:
   - amount spent / cost / expenditure -> ARPU, revenue, OpEx
   - consumer base / customer base -> subscriber count, user metrics
   - operation cost / operational cost -> OPEX, operational metrics
   - income / earnings / profit -> Revenue, EBITDA, PAT

4. TABLE DATA: Extract and present specific row/column values from tables.

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

- YES -> if at least one chunk is about the same entity, metric, time period, or topic.
- NO -> if the chunks are about completely different entities, topics, or metrics.

Return JSON: {{"relevant": true/false}}"""

# ---------------------------------------------------------------------------
# Prompts — Hybrid (merge SQL + document context)
# ---------------------------------------------------------------------------
HYBRID_SYNTHESIZE_PROMPT = """You are an expert financial document assistant with access to BOTH
structured SQL data and unstructured document chunks.

You will receive:
- A user's question
- SQL query results (structured financial data)
- Document chunks (narrative, analysis, tables from PDFs)

Provide the BEST possible answer combining BOTH sources.

## Rules
1. Use SQL data for precise numbers, totals, comparisons, percentages.
2. Use document chunks for explanations, context, management commentary.
3. If SQL and documents agree, state the fact and add context.
4. If they differ, note the discrepancy and provide both perspectives.
5. Present numbers with proper formatting (INR Crore notation).
6. Always mention the time period.
7. Use bullet points for multi-metric answers.

## SQL Results
{sql_results}

## Document Chunks
{doc_context}

## User Question
{query}

Return your answer as a JSON object with exactly these keys:
  - "answer": your synthesized response (string)
  - "metadata": []

Return ONLY the JSON object. Do NOT wrap it in markdown code fences."""

# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------
class RAGState(TypedDict):
    """State that flows through every node in the graph."""
    messages: Annotated[list, add_messages]
    query: str
    # Routing
    route: str                          # "sql", "document", or "hybrid"
    # SQL path
    sql_query: str
    sql_results: List[Dict[str, Any]]
    sql_answer: str
    sql_row_count: int
    # Document path (original fields, unchanged)
    raw_chunks: List[Dict[str, Any]]
    reranked_chunks: List[Dict[str, Any]]
    metadata: List[Dict[str, Any]]
    # Retry loop
    retries: int
    synthesized_answer: str
    answer_relevant: bool


# ---------------------------------------------------------------------------
# Helper Utilities (original, unchanged)
# ---------------------------------------------------------------------------
TOOL_NAMES = {"vector_search", "fts_search", "hybrid_search"}


def _extract_response_text(content: Any) -> str:
    """Safely extract a plain string from Gemini's response content."""
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
    """Remove markdown code fences that LLMs sometimes wrap around JSON output."""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.index("\n") if "\n" in text else len(text)
        text = text[first_newline:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        return text.strip()
    return text


def _extract_text_from_chunk(chunk: Dict[str, Any]) -> str:
    """Extract searchable text from a chunk."""
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
    """Extract and normalize metadata from a chunk."""
    if isinstance(chunk, dict):
        meta = chunk.get("metadata", {})
        if isinstance(meta, dict) and "metadata" in meta:
            meta = {**meta, **meta["metadata"]}
    else:
        meta = {}

    # Recover fields from JSON content when metadata is missing
    content = chunk.get("content", "") or ""
    if isinstance(content, str) and content.startswith("{"):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                for field in ["llm_generated_caption", "table_data", "docling_caption"]:
                    if parsed.get(field) and not meta.get(field):
                        meta[field] = parsed[field]
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
    """Walk messages backwards and return raw chunk dicts from tool messages."""
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


def _build_metadata_entry(
    chunk: Dict[str, Any],
    rank: int,
    scores: Dict[str, Any],
) -> Dict[str, Any]:
    """Build one metadata dict that merges scores, rerank scores, and ingestion metadata."""
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


def _build_sql_metadata_entry(
    sql_query: str,
    sql_answer: str,
    sql_row_count: int,
    route: str,
    query: str,
) -> Dict[str, Any]:
    """
    Build a synthetic metadata entry for SQL results.

    This is a MetadataEntry-compatible dict with extra fields.
    Respects the existing schema constraints:
      - rank: ge=1  (we use 1)
      - modality: Literal["text", "table", "image"]  (we use "text")
    SQL-specific data is carried as extra fields (extra="allow"):
      - route: "sql" or "hybrid"
      - sql_query: the generated SQL
      - sql_answer: the LLM-synthesized answer from SQL
      - sql_row_count: number of rows returned
    Consumers can check entry.route to distinguish SQL from document entries.
    """
    return {
        # Core MetadataEntry fields (must satisfy schema constraints)
        "rank": 1,
        "modality": "text",
        "content": sql_answer,
        "citation": sql_answer[:500] if sql_answer else "",
        # Extra fields (allowed by MetadataEntry.Config.extra = "allow")
        "route": route,
        "sql_query": sql_query,
        "sql_answer": sql_answer,
        "sql_row_count": sql_row_count,
        "section_header": f"SQL Query ({route} path)",
        "document_name": "financial_database",
    }


def _format_chunks_as_context(chunks: List[Dict[str, Any]], max_chunks: int = 5) -> str:
    """Format top chunks into a readable context block for the LLM synthesizer."""
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
                f"[Table {i} - {doc}, p.{page}, Section: {section}]\n"
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
                f"[Chart/Figure {i} - {doc}, p.{page}, Section: {section}]\n"
                f"Description: {caption}"
            )
        else:
            text_preview = text[:400] + "..." if len(text) > 400 else text
            parts.append(
                f"[Text {i} - {doc}, p.{page}, Section: {section}]\n"
                f"{text_preview}"
            )

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# SQL Helper Functions
# ---------------------------------------------------------------------------
def _execute_sql_query(sql: str) -> tuple:
    """Execute SQL via psycopg2 and return (columns, rows)."""
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


def _format_sql_results(columns: list, rows: list, max_rows: int = 20) -> str:
    """Format SQL results as a readable table string."""
    if not columns or not rows:
        return "(no results)"
    header = " | ".join(columns)
    separator = "-+-".join(["---"] * len(columns))
    row_lines = []
    for row in rows[:max_rows]:
        vals = [str(row.get(c, "")) for c in columns]
        row_lines.append(" | ".join(vals))
    if len(rows) > max_rows:
        row_lines.append(f"... ({len(rows) - max_rows} more rows)")
    return f"{header}\n{separator}\n" + "\n".join(row_lines)


# ============================================================================
# Graph Nodes
# ============================================================================

# ── Router ──────────────────────────────────────────────────────────────
def router_node(state: RAGState) -> dict:
    """
    LLM-based query router. Classifies the user query into one of:
      - "sql": structured data query (realtime data, user data of jio, shareholders data of reliance)
      - "document": unstructured content query (narratives, explanations)
      - "hybrid": needs both SQL data and document context
    """
    query = state["query"]

    response = llm.invoke([
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ])

    raw = _extract_response_text(response.content)
    raw = _strip_json_fences(raw)

    route = "document"  # default fallback
    try:
        parsed = json.loads(raw)
        route = parsed.get("route", "document").lower().strip()
    except (json.JSONDecodeError, TypeError):
        logger.warning("router_node: failed to parse route, defaulting to document: %s", raw)

    # Validate
    if route not in ("sql", "document", "hybrid"):
        logger.warning("router_node: invalid route '%s', defaulting to document", route)
        route = "document"

    logger.info("router_node: query='%s' -> route=%s | thread_id=%s",query[:80], route, state.get("messages", [{}])[-1].__dict__.get("id", "N/A")[:8])
    return {"route": route}


# ── SQL Path Nodes ──────────────────────────────────────────────────────
def sql_write_node(state: RAGState) -> dict:
    """Generate a SQL query from the user's natural language question."""
    query = state["query"]

    if not _DB_SCHEMA:
        logger.error("sql_write_node: no DB schema available")
        return {"sql_query": "", "sql_results": [], "sql_row_count": 0}

    prompt = ChatPromptTemplate.from_template(SQL_QUERY_WRITER_PROMPT)
    chain = prompt | llm | StrOutputParser()
    raw_sql = chain.invoke({"schema": _DB_SCHEMA, "question": query})

    # Strip markdown fences
    sql = raw_sql.strip()
    if sql.startswith("```"):
        lines = sql.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        sql = "\n".join(lines).strip()

    logger.info("sql_write_node: SQL=%s", sql[:200])
    return {"sql_query": sql}


def sql_execute_node(state: RAGState) -> dict:
    """Execute the generated SQL query and store results."""
    sql = state["sql_query"]

    if not sql:
        logger.warning("sql_execute_node: no SQL to execute")
        return {"sql_results": [], "sql_row_count": 0}

    try:
        columns, rows = _execute_sql_query(sql)
        logger.info(
            "sql_execute_node: thread_id=%s | SQL executed | %d rows | query='%s'",
            state.get("messages", [{}])[-1].__dict__.get("id", "N/A")[:8],
            len(rows), sql[:100]
        )
        return {"sql_results": rows, "sql_row_count": len(rows)}
    except Exception as exc:
        logger.error("sql_execute_node: execution failed: %s | SQL: %s", exc, sql[:200])
        return {"sql_results": [], "sql_row_count": 0}


def sql_synthesize_node(state: RAGState) -> dict:
    """
    Synthesize a natural language answer from SQL results.

    Stores the answer in synthesized_answer AND builds a metadata entry
    with the SQL data (sql_query, sql_answer, sql_row_count) that will
    be picked up in run_rag_agent().
    """
    query = state["query"]
    sql = state["sql_query"]
    rows = state.get("sql_results", [])
    route = state.get("route", "sql")

    if not rows:
        answer = (
            "No data was found in the database for this query. "
            "The financial tables may not contain the specific metrics requested. "
            "Try rephrasing with different financial terms."
        )
        sql_meta = _build_sql_metadata_entry(
            sql_query=sql,
            sql_answer=answer,
            sql_row_count=0,
            route=route,
            query=query,
        )
        return {
            "sql_answer": answer,
            "synthesized_answer": answer,
            "metadata": [sql_meta],
        }

    # Build a readable results table
    columns = list(rows[0].keys()) if rows else []
    results_text = _format_sql_results(columns, rows)

    prompt = ChatPromptTemplate.from_template(SQL_ANSWER_SYNTHESIZER_PROMPT)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "question": query,
        "sql_query": sql,
        "results": results_text,
    })

    answer = answer.strip()
    logger.info("sql_synthesize_node: answer generated (%d chars)", len(answer))

    # Build SQL metadata entry with the answer + query
    sql_meta = _build_sql_metadata_entry(
        sql_query=sql,
        sql_answer=answer,
        sql_row_count=len(rows),
        route=route,
        query=query,
    )

    return {
        "sql_answer": answer,
        "synthesized_answer": answer,
        "metadata": [sql_meta],
    }


# ── Document Path Nodes (original, unchanged) ──────────────────────────
def agent_node(state: RAGState) -> dict:
    """First LLM call - decides which retrieval tool to invoke."""
    messages = state["messages"]
    query = state["query"]
    retries = state.get("retries", 0)

    system_prompt = AGENT_RETRY_SYSTEM_PROMPT if retries > 0 else AGENT_SYSTEM_PROMPT
    system = SystemMessage(content=system_prompt)
    user = HumanMessage(content=query)

    if retries > 0:
        response = llm_with_tools.invoke([system, user])
    else:
        response = llm_with_tools.invoke([system] + messages + [user])

    logger.info("agent_node (retry=%d): tool_calls=%s", retries, response.tool_calls)
    return {"messages": [response]}


def extract_chunks_node(state: RAGState) -> dict:
    """Extract raw chunks from the latest ToolMessage."""
    chunks = _extract_chunks_from_messages(state["messages"])
    logger.info("extract_chunks: found %d raw chunks", len(chunks))
    return {"raw_chunks": chunks}


def rerank_node(state: RAGState) -> dict:
    """Takes raw chunks and reranks them via Cohere."""
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
                "rerank_node: %d -> %d chunks (model=%s)",
                len(raw_chunks), len(reranked), RERANK_MODEL,
            )
            return {"reranked_chunks": reranked}
        except Exception as exc:
            logger.warning("rerank_node failed, using original order: %s", exc)

    return {"reranked_chunks": raw_chunks}


def score_node(state: RAGState) -> dict:
    """Computes programmatic scores for every reranked chunk."""
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
    """Second LLM call - generates the final answer using reranked & scored chunks."""
    chunks = state["reranked_chunks"]
    query = state["query"]

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

    logger.info(
        "synthesize_node: thread_id=%s | answer generated | route=%s | chunks_used=%d",
        state.get("messages", [{}])[-1].__dict__.get("id", "N/A")[:8],
        state.get("route"),
        len(state.get("reranked_chunks", []))
    )

    clean_message = AIMessage(content=answer_text)
    return {
        "messages": [clean_message],
        "synthesized_answer": answer_text,
    }


def answer_relevance_node(state: RAGState) -> dict:
    """Check whether the retrieved chunks are relevant to the user's query."""
    query = state["query"]
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

    prompt = CHUNK_RELEVANCE_PROMPT.format(query=query, chunks_text=chunks_text)

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
        logger.warning("answer_relevance_node: failed to parse: %s", result_text)
        relevant = True  # assume relevant to avoid infinite loops

    logger.info(
        "answer_relevance_node: query='%s' | %d chunks | relevant=%s",
        query[:60], len(chunks), relevant,
    )
    return {"answer_relevant": relevant}


def query_reformulate_node(state: RAGState) -> dict:
    """Rewrite the query for better document retrieval."""
    query = state["query"]
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
        "query_reformulate_node: attempt %d/%d - '%s' -> '%s'",
        new_retries, MAX_RETRIES, query, reformulated,
    )

    context_msg = HumanMessage(
        content=(
            f"[Query Reformulation - attempt {new_retries} of {MAX_RETRIES}] "
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


# ── Hybrid Path Nodes ───────────────────────────────────────────────────
def hybrid_merge_node(state: RAGState) -> dict:
    """
    Hybrid path: both SQL results and document chunks are now available.
    This node injects the SQL metadata entry into the metadata list
    (alongside the document chunk entries from score_node).
    """
    sql_query = state.get("sql_query", "")
    sql_answer = state.get("sql_answer", "")
    sql_row_count = state.get("sql_row_count", 0)
    route = state.get("route", "hybrid")
    query = state["query"]

    # Build SQL metadata entry
    sql_meta = _build_sql_metadata_entry(
        sql_query=sql_query,
        sql_answer=sql_answer,
        sql_row_count=sql_row_count,
        route=route,
        query=query,
    )

    # Prepend SQL entry to existing document metadata (if any)
    doc_metadata = state.get("metadata", [])
    merged = [sql_meta] + doc_metadata

    logger.info(
        "hybrid_merge_node: %d SQL rows + %d doc entries -> %d total metadata",
        sql_row_count, len(doc_metadata), len(merged),
    )
    return {"metadata": merged}


def hybrid_synthesize_node(state: RAGState) -> dict:
    """
    Synthesize a combined answer from both SQL data and document chunks.
    Used in the hybrid path.
    """
    query = state["query"]
    sql = state.get("sql_query", "")
    rows = state.get("sql_results", [])
    chunks = state.get("reranked_chunks", [])

    # Format SQL results
    sql_columns = list(rows[0].keys()) if rows else []
    sql_results_text = _format_sql_results(sql_columns, rows) if rows else "(no SQL results)"

    # Format document chunks
    doc_context = _format_chunks_as_context(chunks) if chunks else "(no document chunks found)"

    system = SystemMessage(content=HYBRID_SYNTHESIZE_PROMPT)
    user = HumanMessage(
        content=HYBRID_SYNTHESIZE_PROMPT.format(
            sql_results=sql_results_text,
            doc_context=doc_context,
            query=query,
        )
    )

    response = llm.invoke([system, user])
    answer_text = _extract_response_text(response.content)
    answer_text = _strip_json_fences(answer_text)

    logger.info("hybrid_synthesize_node: combined answer generated (%d chars)", len(answer_text))

    clean_message = AIMessage(content=answer_text)
    return {
        "messages": [clean_message],
        "synthesized_answer": answer_text,
    }


# ============================================================================
# Conditional Edge Functions
# ============================================================================
def _route_query(state: RAGState) -> str:
    """
    After the router node, decide which path to take.
    Returns "sql", "document", or "hybrid".
    """
    route = state.get("route", "document")
    logger.info("_route_query: %s", route)
    return route


def _has_relevant_chunks(state: RAGState) -> str:
    """After extract_chunks, decide what to do next."""
    chunks = state.get("raw_chunks", [])
    retries = state.get("retries", 0)

    if chunks:
        return "rerank"
    if retries < MAX_RETRIES:
        logger.info("No chunks (retries=%d/%d) -> reformulate", retries, MAX_RETRIES)
        return "reformulate"

    logger.warning("No chunks, retries exhausted (%d/%d) -> fallback synthesize", retries, MAX_RETRIES)
    return "synthesize"


def _check_answer_relevance(state: RAGState) -> str:
    """After evaluate_relevance, decide whether to accept or retry."""
    relevant = state.get("answer_relevant", False)
    retries = state.get("retries", 0)

    if relevant:
        logger.info("Answer is relevant -> END")
        return "end"

    if retries < MAX_RETRIES:
        logger.info("Answer NOT relevant (retries=%d/%d) -> reformulate", retries, MAX_RETRIES)
        return "reformulate"

    logger.warning("Answer NOT relevant, retries exhausted (%d/%d) -> END", retries, MAX_RETRIES)
    return "end"

def inspect_checkpoints(thread_id: str, max_steps: int = 10) -> List[Dict[str, Any]]:
    """
    Debug helper: retrieve and print checkpoint history for a thread_id.
    
    Returns list of checkpoint summaries for inspection.
    """
    if rag_checkpointer is None:
        logger.error("inspect_checkpoints: no checkpointer available")
        return []
    
    checkpoints = []
    config = {"configurable": {"thread_id": thread_id}}
    
    # Get latest checkpoint
    checkpoint_tuple = rag_checkpointer.get_tuple(config)
    if not checkpoint_tuple:
        logger.warning("inspect_checkpoints: no checkpoints found for %s", thread_id)
        return []
    
    checkpoint, checkpoint_config = checkpoint_tuple
    
    # Walk back through parent checkpoints
    current_config = checkpoint_config
    for step in range(max_steps):
        cp_tuple = rag_checkpointer.get_tuple(current_config)
        if not cp_tuple:
            break
            
        cp, cp_cfg = cp_tuple
        checkpoints.append({
            "step": step,
            "checkpoint_id": cp_cfg["checkpoint_id"],
            "parent_id": cp["parent_config"]["checkpoint_id"] if cp["parent_config"] else None,
            "route": cp["channel_values"].get("route"),
            "query_preview": cp["channel_values"].get("query", "")[:80],
            "metadata_count": len(cp["channel_values"].get("metadata", [])),
            "timestamp": cp["metadata"].get("written_at"),
        })
        
        # Move to parent
        if cp["parent_config"]:
            current_config = {
                "configurable": {
                    "thread_id": cp["parent_config"]["thread_id"],
                    "checkpoint_id": cp["parent_config"]["checkpoint_id"],
                    "checkpoint_ns": cp["parent_config"].get("checkpoint_ns", ""),
                }
            }
        else:
            break
    
    # Log summary
    logger.info(
        "inspect_checkpoints: thread_id=%s | found %d checkpoints",
        thread_id, len(checkpoints)
    )
    for cp in checkpoints:
        logger.debug("  Step %d: route=%s, query='%s...', metadata=%d", 
                    cp["step"], cp["route"], cp["query_preview"], cp["metadata_count"])
    
    return checkpoints


# ============================================================================
# Graph Construction
# ============================================================================
def build_graph() -> StateGraph:
    graph = StateGraph(RAGState)

    # ── Nodes ────────────────────────────────────────────────────────────
    graph.add_node("router", router_node)
    graph.add_node("sql_write", sql_write_node)
    graph.add_node("sql_execute", sql_execute_node)
    graph.add_node("sql_synthesize", sql_synthesize_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools=RETRIEVAL_TOOLS))
    graph.add_node("extract_chunks", extract_chunks_node)
    graph.add_node("query_reformulate", query_reformulate_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("score", score_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("evaluate_relevance", answer_relevance_node)
    graph.add_node("hybrid_merge", hybrid_merge_node)
    graph.add_node("hybrid_synthesize", hybrid_synthesize_node)

    # ── Edges ────────────────────────────────────────────────────────────
    graph.add_edge(START, "router")

    # Router dispatches to 3 paths
    graph.add_conditional_edges(
        "router",
        _route_query,
        {
            "sql": "sql_write",
            "document": "agent",
            "hybrid": "sql_write",
        },
    )

    # ── SQL path: sql_write -> sql_execute -> (conditional based on route) ──────────
    graph.add_edge("sql_write", "sql_execute")
    
    # FIX #1: Single conditional edge that checks route
    graph.add_conditional_edges(
        "sql_execute",
        lambda state: "sql_synthesize" if state.get("route") == "sql" else "agent",
        {
            "sql_synthesize": "sql_synthesize",
            "agent": "agent",
        },
    )
    
    graph.add_edge("sql_synthesize", END)

    # ── Document/Hybrid path: agent -> tools -> extract ──────────────────
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

    # FIX #2: Single conditional edge for score node
    graph.add_conditional_edges(
        "score",
        lambda state: "hybrid_merge" if state.get("route") == "hybrid" else "synthesize",
        {
            "synthesize": "synthesize",
            "hybrid_merge": "hybrid_merge",
        },
    )
    # REMOVED: graph.add_edge("score", "synthesize")  <-- was duplicate!

    # FIX #3: Conditional edge for synthesize based on route
    graph.add_conditional_edges(
        "synthesize",
        lambda state: "evaluate_relevance" if state.get("route") == "document" else END,
        {
            "evaluate_relevance": "evaluate_relevance",
            END: END,
        },
    )
    # REMOVED: graph.add_edge("synthesize", "evaluate_relevance")  <-- was duplicate!
    # REMOVED: graph.add_edge("synthesize", END)  <-- was duplicate!

    graph.add_conditional_edges(
        "evaluate_relevance",
        _check_answer_relevance,
        {
            "end": END,
            "reformulate": "query_reformulate",
        },
    )

    # ── Hybrid path: merge -> synthesize -> END ──────────────────────────
    graph.add_edge("hybrid_merge", "hybrid_synthesize")
    graph.add_edge("hybrid_synthesize", END) 
    # graph = graph.compile()
    # img = graph.get_graph().draw_mermaid_png()
    # with open("graph_img_eg_10.png", "wb") as f:
    #     f.write(img)
    # print("✓ Graph saved to graph_img_eg_10.png")
    checkpointer = MemorySaver()  # In-memory (dev); swap for PostgresSaver in prod
    compiled_graph = graph.compile(checkpointer=checkpointer)
    
    logger.info("✓ Graph compiled with MemorySaver checkpointing enabled")
    return compiled_graph, checkpointer  # ← Return both!
    


# ---------------------------------------------------------------------------
# Compiled Graph Singleton
# ---------------------------------------------------------------------------
rag_graph, rag_checkpointer = build_graph() 


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def run_rag_agent(
    query: str, 
    k: int = 10,
    thread_id: Optional[str] = None  # ← NEW: conversation ID
) -> QueryResponse:
    """
    Stateless RAG + SQL agent entry point with checkpointing support.
    
    Args:
        query: User's natural language question.
        k: Number of top-k chunks to retrieve.
        thread_id: Unique conversation ID for history persistence. 
                   If None, generates a new UUID per call (stateless).
    """
    import uuid
    from langchain_core.messages import HumanMessage
    
    # Generate thread_id if not provided (stateless fallback)
    if thread_id is None:
        thread_id = f"stateless_{uuid.uuid4().hex[:8]}"
        logger.debug("run_rag_agent: generated stateless thread_id=%s", thread_id)
    
    config = {"configurable": {"thread_id": thread_id}}
    logger.info("run_rag_agent: thread_id=%s, query='%s'", thread_id, query[:100])
    
    initial_state: RAGState = {
        "messages": [HumanMessage(content=query)],  # ← Start with user message
        "query": query,
        "route": "document",
        "sql_query": "",
        "sql_results": [],
        "sql_answer": "",
        "sql_row_count": 0,
        "raw_chunks": [],
        "reranked_chunks": [],
        "metadata": [],
        "retries": 0,
        "synthesized_answer": "",
        "answer_relevant": False,
    }

    # Invoke with config → checkpoints saved automatically after each node
    final_state = rag_graph.invoke(initial_state, config=config)
    
    logger.info(
        "run_rag_agent: completed | thread_id=%s | route=%s | answer_len=%d",
        thread_id,
        final_state.get("route"),
        len(final_state.get("synthesized_answer", ""))
    )

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

    # For SQL-only path, answer is stored in state directly (no AIMessage in messages)
    if not messages and final_state.get("synthesized_answer"):
        answer = final_state["synthesized_answer"]

    # Build MetadataEntry objects from dicts
    # MetadataEntry has extra="allow" so SQL fields (sql_query, sql_answer, etc.)
    # pass validation seamlessly
    metadata_entries: List[MetadataEntry] = []
    for meta_dict in metadata:
        try:
            metadata_entries.append(MetadataEntry(**meta_dict))
        except Exception as exc:
            logger.warning("run_rag_agent: failed to build MetadataEntry: %s | dict=%s", exc, list(meta_dict.keys()))
            # Fallback: skip invalid entries
            continue

    return QueryResponse(
        query=query,
        answer=answer,
        metadata=metadata_entries,
    )