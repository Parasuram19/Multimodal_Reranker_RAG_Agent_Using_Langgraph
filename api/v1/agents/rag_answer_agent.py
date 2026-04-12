# """
# LangGraph-based RAG Agent for Insurance & Policy Q&A.
# Compatible with ingestion.py metadata structure.

# Graph architecture:
#     START → agent_node → tools_condition → [tool execution]
#         → extract_chunks → [has chunks?]
#             ├─ Yes → rerank → score → synthesize → END
#             └─ No + retries < max → query_reformulate → agent_node (loop)
#             └─ No + retries exhausted → synthesize (fallback) → END

# All scoring is programmatic (TF-IDF/BM25). Reranking via Cohere is optional.
# """

# import json
# import logging
# import os
# from typing import Annotated, Any, Dict, List, Optional, TypedDict, Union

# from dotenv import load_dotenv
# from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
# from langchain_core.tools import BaseTool
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langgraph.graph import END, START, StateGraph
# from langgraph.graph.message import add_messages
# from langgraph.prebuilt import ToolNode, tools_condition

# from api.v1.schemas.query_schema import QueryResponse, MetadataEntry
# from api.v1.tools.fts_search_tool import fts_search
# from api.v1.tools.hybrid_search_tool import hybrid_search
# from api.v1.tools.vector_search_tool import vector_search
# from api.v1.utils.reranker import rerank_chunks
# from api.v1.utils.scoring import score_chunks

# load_dotenv(override=True)
# logger = logging.getLogger(__name__)

# # ---------------------------------------------------------------------------
# # Config
# # ---------------------------------------------------------------------------
# ENABLE_RERANK: bool = os.getenv("ENABLE_RERANK", "true").lower() == "true"
# RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", "5"))
# RERANK_MODEL: str = os.getenv("RERANK_MODEL", "rerank-v3.5")
# MAX_QUERY_RETRIES: int = int(os.getenv("MAX_QUERY_RETRIES", "2"))

# # ---------------------------------------------------------------------------
# # LLM — bound with retrieval tools
# # ---------------------------------------------------------------------------
# MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
# GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# if not GOOGLE_API_KEY:
#     raise ValueError("GOOGLE_API_KEY environment variable is required")

# llm = ChatGoogleGenerativeAI(
#     model=MODEL_NAME,
#     google_api_key=GOOGLE_API_KEY,
#     temperature=0,
# )

# RETRIEVAL_TOOLS: List[BaseTool] = [vector_search, fts_search, hybrid_search]
# llm_with_tools = llm.bind_tools(RETRIEVAL_TOOLS)

# # ---------------------------------------------------------------------------
# # Prompts
# # ---------------------------------------------------------------------------
# AGENT_SYSTEM_PROMPT = """You are an expert Financial Document assistant.

# You have access to three retrieval tools:
#   • vector_search  → best for natural language / conceptual questions
#   • fts_search     → best for codes, IDs, abbreviations, exact keywords  
#   • hybrid_search  → best for short or ambiguous queries

# Rules:
# 1. Choose exactly ONE tool based on the query type.
# 2. Call the tool with the ORIGINAL user query (do not modify it).
# 3. Return ONLY a tool call — do not answer the question yourself.
# 4. The tool will return document chunks with metadata including:
#    - modality: "text", "table", or "image"
#    - section_header: the document section this chunk belongs to
#    - For tables: table_data, docling_caption, llm_generated_caption
#    - For images: llm_generated_caption describing the chart/figure
# 5. After retrieval, you will synthesize an answer using ONLY the returned chunks."""


# AGENT_RETRY_SYSTEM_PROMPT = """You are an expert Financial Document assistant.

# The previous retrieval attempt returned NO relevant document chunks.

# You have access to three retrieval tools:
#   • vector_search  → best for natural language / conceptual questions
#   • fts_search     → best for codes, IDs, abbreviations, exact keywords
#   • hybrid_search  → best for short or ambiguous queries

# Rules:
# 1. Choose exactly ONE tool and call it with the REFORMULATED query 
#    provided in the most recent human message.
# 2. You MUST return a tool call — do not answer the question yourself.
# 3. The reformulated query is optimized for document retrieval."""


# QUERY_REFORMULATE_SYSTEM_PROMPT = """You are a query rewriting specialist for a
# Financial Document retrieval system.

# The user's original query returned NO relevant document chunks.
# Produce a more structured, retrieval-friendly version that maximizes
# matching relevant financial documents.

# Guidelines:
# 1. Replace vague pronouns with specific entities from context.
# 2. Use precise financial terminology: "revenue", "EBITDA", "PAT",
#    "ARPU", "OPEX", "margin", "YoY growth", "consolidated", "segment".
# 3. Break compound questions into a single focused query.
# 4. Include relevant financial metrics, section names, or time periods
#    (e.g., "Q2 FY25", "2Q FY24", "H1 FY25").
# 5. If conversational/ambiguous, rephrase as concise keyword-style search.
# 6. Expand ambiguous terms with financial synonyms:
#    - "amount spent / cost" → "expenditure", "OPEX", "spending"
#    - "consumer base" → "customer base", "subscriber count", "user metrics"
#    - "profit" → "EBITDA", "PAT", "net income", "bottom-line"
#    - "income" → "revenue", "top-line", "turnover"
#    - "operation cost" → "OPEX", "operating expense", "operational cost"
# 7. Return ONLY the reformulated query string — nothing else."""

# SYNTHESIZE_SYSTEM_PROMPT = """You are an expert financial document assistant.

# You will be given document chunks and a user question. Provide the BEST
# possible answer using the available chunks.

# ## Rules

# 1. DIRECT MATCH: If a chunk directly answers the question, use it.

# 2. PARTIAL / RELATED MATCH (CRITICAL): If no chunk has the exact answer
#    BUT contains RELATED data, you MUST provide that related data and
#    explain how it connects. Do NOT say "no information found" when
#    related metrics exist. Examples:
#    - User asks "amount spent on consumers" → provide ARPU, customer count,
#      revenue per user from available tables
#    - User asks "operation cost" → provide operational metrics, OpEx data,
#      or the closest available financial metrics
#    - User asks "profit" → provide EBITDA, PAT, or margin data

# 3. FINANCIAL TERM MAPPING — treat these as equivalent:
#    - amount spent / cost / expenditure → ARPU, revenue, OpEx
#    - consumer base / customer base → subscriber count, user metrics
#    - operation cost / operational cost → OPEX, operational metrics
#    - income / earnings / profit → Revenue, EBITDA, PAT

# 4. TABLE DATA: Extract and present specific row/column values from tables.
#    Tables often contain the most precise answers — never ignore them.

# 5. ONLY say "the documents do not contain this information" when ZERO
#    chunks are even tangentially related to the question.

# Return your answer as a JSON object with exactly these keys:
#   - "answer": your synthesized response (string)
#   - "metadata": []

# Return ONLY the JSON object. Do NOT wrap it in markdown code fences."""

# # ---------------------------------------------------------------------------
# # Graph State
# # ---------------------------------------------------------------------------
# class RAGState(TypedDict):
#     """State that flows through every node in the graph."""
#     messages: Annotated[list, add_messages]
#     query: str
#     raw_chunks: List[Dict[str, Any]]
#     reranked_chunks: List[Dict[str, Any]]
#     metadata: List[Dict[str, Any]]
#     retries: int


# # ---------------------------------------------------------------------------
# # Helper Utilities
# # ---------------------------------------------------------------------------
# TOOL_NAMES = {"vector_search", "fts_search", "hybrid_search"}

# # ... [keep all imports and config from previous version] ...

# # ---------------------------------------------------------------------------
# # Helper Utilities (UPDATED)
# # ---------------------------------------------------------------------------

# def _extract_text_from_chunk(chunk: Dict[str, Any]) -> str:
#     """
#     Extract searchable text from a chunk, handling:
#     - Plain string content
#     - JSON string content (tables/images from ingestion.py)
#     - List content (from Gemini embeddings with JSONB)
#     - Dict content (structured metadata)
    
#     Compatible with Gemini embeddings storing content as list/dict.
#     """
#     content = chunk.get("content")
    
#     # Handle None
#     if content is None:
#         return ""
    
#     # Handle list → join elements
#     if isinstance(content, list):
#         parts = []
#         for item in content:
#             if isinstance(item, dict):
#                 for key in ["content", "text", "table_data", "llm_generated_caption", "docling_caption"]:
#                     if key in item and isinstance(item[key], str):
#                         parts.append(item[key])
#             elif item is not None:
#                 parts.append(str(item))
#         return " ".join(parts)
    
#     # Handle dict → extract text fields
#     if isinstance(content, dict):
#         parts = []
#         if content.get("type") == "table":
#             parts.extend([
#                 content.get("section_header", ""),
#                 content.get("docling_caption", ""),
#                 content.get("llm_generated_caption", ""),
#                 content.get("table_data", ""),
#             ])
#         elif content.get("type") == "image":
#             parts.extend([
#                 content.get("section_header", ""),
#                 content.get("docling_caption", ""),
#                 content.get("llm_generated_caption", ""),
#             ])
#         else:
#             for key in ["content", "text", "chunk_text", "document", "caption"]:
#                 if key in content and isinstance(content[key], str):
#                     parts.append(content[key])
#         return " ".join(p for p in parts if p)
    
#     # Handle string → parse JSON if applicable
#     if isinstance(content, str):
#         content = content.strip()
#         if not content:
#             return ""
#         if content.startswith("{") or content.startswith("["):
#             try:
#                 parsed = json.loads(content)
#                 return _extract_text_from_chunk({"content": parsed})  # Recursive
#             except (json.JSONDecodeError, TypeError):
#                 pass
#         return content
    
#     return str(content)


# def _extract_response_text(content: Any) -> str:
#     """
#     Safely extract a plain string from Gemini's response content.

#     Gemini returns content as a list of dicts, e.g.:
#         [{"text": "the answer"}]
#     but nodes expect a plain string to call .strip(), json.loads(), etc.
#     """
#     if isinstance(content, str):
#         return content

#     if isinstance(content, list):
#         # Gemini format: [{"text": "..."}, ...]
#         parts = []
#         for item in content:
#             if isinstance(item, dict) and "text" in item:
#                 parts.append(item["text"])
#             elif isinstance(item, str):
#                 parts.append(item)
#             else:
#                 parts.append(str(item))
#         return "".join(parts)

#     if isinstance(content, dict) and "text" in content:
#         return content["text"]

#     return str(content) if content else ""
# def _get_chunk_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Extract and normalize metadata from a chunk.
#     Handles nested metadata structures from PGVector JSONB.
#     Also recovers fields from JSON content when metadata is missing.
#     """
#     if isinstance(chunk, dict):
#         meta = chunk.get("metadata", {})
#         if isinstance(meta, dict) and "metadata" in meta:
#             meta = {**meta, **meta["metadata"]}
#     else:
#         meta = {}

#     # FIX: If llm_generated_caption or table_data is missing from metadata,
#     # try to recover them from the JSON content string
#     content = chunk.get("content", "") or ""
#     if isinstance(content, str) and content.startswith("{"):
#         try:
#             parsed = json.loads(content)
#             if isinstance(parsed, dict):
#                 if parsed.get("llm_generated_caption") and not meta.get("llm_generated_caption"):
#                     meta["llm_generated_caption"] = parsed["llm_generated_caption"]
#                 if parsed.get("table_data") and not meta.get("table_data"):
#                     meta["table_data"] = parsed["table_data"]
#                 if parsed.get("docling_caption") and not meta.get("docling_caption"):
#                     meta["docling_caption"] = parsed["docling_caption"]
#         except (json.JSONDecodeError, TypeError):
#             pass

#     normalized = {
#         "document_id": meta.get("document_id"),
#         "document_name": meta.get("document_name") or meta.get("source_file"),
#         "page_number": meta.get("page") or meta.get("page_number"),
#         "chunk_index": meta.get("chunk_index"),
#         "modality": meta.get("modality", "text"),
#         "element_type": meta.get("element_type"),
#         "section_header": meta.get("section") or meta.get("section_header"),
#         "docling_label": meta.get("docling_label"),
#         "docling_caption": meta.get("docling_caption"),
#         "llm_generated_caption": meta.get("llm_generated_caption"),  # Now recovered
#         "table_data": meta.get("table_data"),                         # Now recovered
#         "table_title": meta.get("table_title"),
#         "bbox": meta.get("bbox"),
#         "source_file": meta.get("source_file"),
#         "ingested_at": meta.get("ingested_at"),
#         "cosine_similarity": None,
#         "bm25": None,
#         "relevance_score": None,
#         "rerank_score": None,
#         "rerank_index": None,
#     }

#     skip_keys = set(normalized.keys())
#     for k, v in meta.items():
#         if k not in skip_keys and v is not None:
#             normalized[k] = v

#     return {k: v for k, v in normalized.items() if v is not None}

# def _extract_chunks_from_messages(messages: list) -> List[Dict[str, Any]]:
#     """
#     Walk messages backwards and return raw chunk dicts from tool messages.
#     Handles content that may be list/dict from Gemini embeddings.
#     """
#     for msg in reversed(messages):
#         if isinstance(msg, ToolMessage) and msg.name in TOOL_NAMES:
#             try:
#                 parsed = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
#                 if isinstance(parsed, list):
#                     return parsed
#                 elif isinstance(parsed, dict):
#                     # Handle single dict or dict with 'results' key
#                     return parsed.get("results", [parsed]) if "results" in parsed else [parsed]
#             except (json.JSONDecodeError, TypeError, AttributeError):
#                 # Fallback: try to use content as-is if it's a list
#                 if isinstance(msg.content, list):
#                     return msg.content
#                 elif isinstance(msg.content, dict):
#                     return [msg.content]
#     return []


# def _strip_json_fences(text: str) -> str:
#     """
#     Remove markdown code fences (```json ... ``` or ``` ... ```)
#     that LLMs sometimes wrap around JSON output.
#     """
#     text = text.strip()
#     # Match ```json ... ``` or ``` ... ```
#     if text.startswith("```"):
#         # Remove opening fence line
#         first_newline = text.index("\n") if "\n" in text else len(text)
#         text = text[first_newline:]
#         # Remove closing fence
#         if text.rstrip().endswith("```"):
#             text = text.rstrip()[:-3]
#         return text.strip()
#     return text

# # ... [rest of rag_answer_agent.py remains the same, using _extract_text_from_chunk] ...
# def _build_metadata_entry(
#     chunk: Dict[str, Any],
#     rank: int,
#     scores: Dict[str, Any],
# ) -> Dict[str, Any]:
#     """
#     Build one metadata dict that merges programmatic scores, rerank scores,
#     and ingestion metadata fields.
#     """
#     meta = _get_chunk_metadata(chunk)
#     text = _extract_text_from_chunk(chunk)
    
#     # Merge scores into metadata
#     entry = {
#         "rank": rank,
#         **scores,  # cosine_similarity, bm25, relevance_score
#         "rerank_score": chunk.get("rerank_score"),
#         "rerank_index": chunk.get("rerank_index"),
#         # Citation/provenance
#         "citation": meta.get("citation", text[:500]),  # First 500 chars as citation
#         "content": text,  # Full searchable text
#         # Ingestion metadata
#         **{k: v for k, v in meta.items() if k not in {
#             "rank", "cosine_similarity", "bm25", "relevance_score",
#             "citation", "content", "rerank_score", "rerank_index"
#         }},
#     }
#     return entry

# def _format_chunks_as_context(chunks: List[Dict[str, Any]], max_chunks: int = 5) -> str:
#     parts = []
#     for i, chunk in enumerate(chunks[:max_chunks], start=1):
#         text = _extract_text_from_chunk(chunk)
#         meta = _get_chunk_metadata(chunk)

#         modality = meta.get("modality", "text")
#         doc = meta.get("document_name", "unknown")
#         page = meta.get("page_number", "N/A")
#         section = meta.get("section_header", "")

#         if modality == "table":
#             caption = (
#                 meta.get("llm_generated_caption")
#                 or meta.get("docling_caption")
#                 or ""
#             )
#             # FIX: Give tables much more space — 1500 chars instead of 300
#             table_preview = text[:1500] + "..." if len(text) > 1500 else text
#             parts.append(
#                 f"[Table {i} — {doc}, p.{page}, Section: {section}]\n"
#                 f"Caption: {caption}\n"
#                 f"{table_preview}"
#             )
#         elif modality == "image":
#             caption = (
#                 meta.get("llm_generated_caption")
#                 or meta.get("docling_caption")
#                 or "No caption"
#             )
#             parts.append(
#                 f"[Chart/Figure {i} — {doc}, p.{page}, Section: {section}]\n"
#                 f"Description: {caption}"
#             )
#         else:
#             text_preview = text[:400] + "..." if len(text) > 400 else text
#             parts.append(
#                 f"[Text {i} — {doc}, p.{page}, Section: {section}]\n"
#                 f"{text_preview}"
#             )

#     return "\n\n---\n\n".join(parts)

# # ---------------------------------------------------------------------------
# # Graph Nodes
# # ---------------------------------------------------------------------------
# def agent_node(state: RAGState) -> dict:
#     """
#     First LLM call — decides which retrieval tool to invoke.
#     Must return a tool_call (no free-text answer).
#     """
#     messages = state["messages"]
#     query = state["query"]
#     retries = state.get("retries", 0)

#     system_prompt = AGENT_RETRY_SYSTEM_PROMPT if retries > 0 else AGENT_SYSTEM_PROMPT
#     system = SystemMessage(content=system_prompt)
#     user = HumanMessage(content=query)

#     response = llm_with_tools.invoke([system] + messages + [user])
#     logger.info("agent_node (retry=%d): tool_calls=%s", retries, response.tool_calls)
#     return {"messages": [response]}


# def rerank_node(state: RAGState) -> dict:
#     """
#     Takes raw chunks from tool output and reranks them via Cohere.
#     Gracefully degrades to original order if disabled or fails.
#     """
#     raw_chunks = state["raw_chunks"]
#     query = state["query"]

#     if not raw_chunks:
#         return {"reranked_chunks": []}

#     if ENABLE_RERANK:
#         try:
#             # Extract text for reranking (handles multimodal content)
#             documents = [_extract_text_from_chunk(c) for c in raw_chunks]
            
#             reranked = rerank_chunks(
#                 query=query,
#                 documents=documents,
#                 original_chunks=raw_chunks,  # Preserve full metadata
#                 top_n=RERANK_TOP_N,
#                 model=RERANK_MODEL,
#             )
#             logger.info(
#                 "rerank_node: %d → %d chunks (model=%s)",
#                 len(raw_chunks), len(reranked), RERANK_MODEL,
#             )
#             return {"reranked_chunks": reranked}
#         except Exception as exc:
#             logger.warning("rerank_node failed, using original order: %s", exc)

#     return {"reranked_chunks": raw_chunks}


# def score_node(state: RAGState) -> dict:
#     """
#     Computes programmatic scores (cosine_similarity, BM25, relevance_score)
#     for every reranked chunk and assembles the final metadata list.
#     """
#     chunks = state["reranked_chunks"]
#     query = state["query"]

#     if not chunks:
#         return {"metadata": []}

#     # Extract text for scoring (handles multimodal content)
#     chunk_texts = [_extract_text_from_chunk(c) for c in chunks]
#     computed_scores = score_chunks(query, chunk_texts)

#     metadata_list = [
#         _build_metadata_entry(chunk, rank=i, scores=scores)
#         for i, (chunk, scores) in enumerate(zip(chunks, computed_scores), start=1)
#     ]

#     logger.info(
#         "score_node: scored %d chunks (best relevance=%.4f)",
#         len(metadata_list),
#         metadata_list[0]["relevance_score"] if metadata_list else 0.0,
#     )
#     return {"metadata": metadata_list}
# def synthesize_node(state: RAGState) -> dict:
#     chunks = state["reranked_chunks"]
#     query = state["query"]

#     if not chunks:
#         fallback_content = json.dumps({
#             "answer": (
#                 "I could not retrieve any relevant documents to answer your "
#                 "question, even after reformulating the query. Please try "
#                 "rephrasing with more specific terms or contact support."
#             ),
#             "metadata": [],
#         })
#         fallback = AIMessage(content=fallback_content)
#         return {"messages": [fallback]}

#     context = _format_chunks_as_context(chunks)

#     system = SystemMessage(content=SYNTHESIZE_SYSTEM_PROMPT)
#     user = HumanMessage(
#         content=f"User question:\n{query}\n\nDocument chunks:\n{context}"
#     )

#     response = llm.invoke([system, user])

#     # Extract text from Gemini's list-of-dicts content
#     answer_text = _extract_response_text(response.content)

#     # FIX: strip markdown fences as safety net
#     answer_text = _strip_json_fences(answer_text)

#     logger.info("synthesize_node: answer generated (%d chars)", len(answer_text))

#     # Re-wrap as clean AIMessage with plain string content
#     clean_message = AIMessage(content=answer_text)
#     return {"messages": [clean_message]}

# def query_reformulate_node(state: RAGState) -> dict:
#     query = state["query"]
#     retries = state.get("retries", 0)
#     new_retries = retries + 1

#     response = llm.invoke([
#         SystemMessage(content=QUERY_REFORMULATE_SYSTEM_PROMPT),
#         HumanMessage(content=query),
#     ])

#     # FIX: handle Gemini's list-of-dicts content format
#     reformulated = _extract_response_text(response.content).strip().strip('"').strip("'")

#     logger.info(
#         "query_reformulate_node: attempt %d/%d — original: '%s' → reformulated: '%s'",
#         new_retries, MAX_QUERY_RETRIES, query, reformulated,
#     )

#     context_msg = HumanMessage(
#         content=(
#             f"[Query Reformulation — attempt {new_retries} of {MAX_QUERY_RETRIES}] "
#             f"Previous retrieval returned no relevant results. "
#             f"Please use this reformulated query for the next retrieval:\n"
#             f"\"{reformulated}\""
#         )
#     )

#     return {
#         "query": reformulated,
#         "retries": new_retries,
#         "messages": [context_msg],
#     }

# def extract_chunks_node(state: RAGState) -> dict:
#     """Extract raw chunks from the latest ToolMessage and store in state."""
#     chunks = _extract_chunks_from_messages(state["messages"])
#     logger.info("extract_chunks: found %d raw chunks", len(chunks))
#     return {"raw_chunks": chunks}


# def _has_relevant_chunks(state: RAGState) -> str:
#     """
#     Conditional edge function called after extract_chunks.
    
#     Returns:
#         "rerank" — chunks found, proceed normally
#         "reformulate" — no chunks and retry budget remains
#         "synthesize" — no chunks and retries exhausted (fallback)
#     """
#     chunks = state.get("raw_chunks", [])
#     retries = state.get("retries", 0)

#     if chunks:
#         return "rerank"
#     if retries < MAX_QUERY_RETRIES:
#         logger.info("No chunks (retries=%d/%d) → reformulate", retries, MAX_QUERY_RETRIES)
#         return "reformulate"
    
#     logger.warning("No chunks, retries exhausted (%d/%d) → fallback", retries, MAX_QUERY_RETRIES)
#     return "synthesize"


# # ---------------------------------------------------------------------------
# # Graph Construction
# # ---------------------------------------------------------------------------
# def build_graph() -> StateGraph:
#     """
#     Build and compile the LangGraph RAG pipeline.
    
#     Pipeline with query-reformulation loop:
#         agent → tools → extract_chunks
#             ├── [chunks] → rerank → score → synthesize → END
#             └── [no chunks] → query_reformulate → agent (loop)
#                   └── [retries exhausted] → synthesize (fallback) → END
#     """
#     graph = StateGraph(RAGState)

#     # Nodes
#     graph.add_node("agent", agent_node)
#     graph.add_node("tools", ToolNode(tools=RETRIEVAL_TOOLS))
#     graph.add_node("extract_chunks", extract_chunks_node)
#     graph.add_node("query_reformulate", query_reformulate_node)
#     graph.add_node("rerank", rerank_node)
#     graph.add_node("score", score_node)
#     graph.add_node("synthesize", synthesize_node)

#     # Edges
#     graph.add_edge(START, "agent")
    
#     graph.add_conditional_edges(
#         "agent",
#         tools_condition,
#         {"tools": "tools", "__end__": END},
#     )
    
#     graph.add_edge("tools", "extract_chunks")
    
#     graph.add_conditional_edges(
#         "extract_chunks",
#         _has_relevant_chunks,
#         {
#             "rerank": "rerank",
#             "reformulate": "query_reformulate",
#             "synthesize": "synthesize",
#         },
#     )
    
#     graph.add_edge("query_reformulate", "agent")  # Loop back with new query
#     graph.add_edge("rerank", "score")
#     graph.add_edge("score", "synthesize")
#     graph.add_edge("synthesize", END)

#     return graph.compile()


# # ---------------------------------------------------------------------------
# # Compiled Graph Singleton
# # ---------------------------------------------------------------------------
# rag_graph = build_graph()


# # ---------------------------------------------------------------------------
# # Main Entry Point
# # ---------------------------------------------------------------------------
# def run_rag_agent(query: str, k: int = 10) -> QueryResponse:
#     initial_state: RAGState = {
#         "messages": [],
#         "query": query,
#         "raw_chunks": [],
#         "reranked_chunks": [],
#         "metadata": [],
#         "retries": 0,
#     }

#     final_state = rag_graph.invoke(initial_state)
#     messages = final_state.get("messages", [])
#     metadata = final_state.get("metadata", [])

#     answer = "I could not generate a proper answer."
#     for msg in reversed(messages):
#         if isinstance(msg, AIMessage) and msg.content:
#             content = _extract_response_text(msg.content).strip()
#             if not content:
#                 continue

#             # FIX: strip markdown fences before parsing
#             content = _strip_json_fences(content)

#             try:
#                 parsed = json.loads(content)
#                 if isinstance(parsed, dict) and "answer" in parsed:
#                     answer = parsed["answer"]
#                     break
#             except (json.JSONDecodeError, TypeError):
#                 answer = content
#                 break

#     return QueryResponse(
#         query=query,
#         answer=answer,
#         metadata=metadata,
#     )

"""
LangGraph-based RAG Agent for Financial Document Q&A.

Graph architecture:
    START → agent_node → tools_condition → [tool execution]
        → extract_chunks → [has chunks?]
            ├─ Yes → rerank → score → synthesize → evaluate_relevance
            │       ├─ relevant → END
            │       └─ not relevant + retries < max → query_reformulate → agent (loop)
            │       └─ not relevant + retries exhausted → END
            └─ No + retries < max → query_reformulate → agent (loop)
            └─ No + retries exhausted → synthesize (fallback) → END

All scoring is programmatic (TF-IDF/BM25). Reranking via Cohere is optional.
Answer relevance is evaluated by a dedicated LLM node before finalizing.
"""

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
from api.v1.utils.reranker import rerank_chunks
from api.v1.utils.scoring import score_chunks

load_dotenv(override=True)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ENABLE_RERANK: bool = os.getenv("ENABLE_RERANK", "true").lower() == "true"
RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", "5"))
RERANK_MODEL: str = os.getenv("RERANK_MODEL", "rerank-v3.5")
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))

# ---------------------------------------------------------------------------
# LLM — bound with retrieval tools
# ---------------------------------------------------------------------------
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is required")

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)

RETRIEVAL_TOOLS: List[BaseTool] = [vector_search, fts_search, hybrid_search]
llm_with_tools = llm.bind_tools(RETRIEVAL_TOOLS)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
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

QUERY_REFORMULATE_SYSTEM_PROMPT = """You are a query rewriting specialist for a
Financial Document retrieval system.

The user's original query returned an answer that was NOT relevant to the
question. Produce a more structured, retrieval-friendly version that maximizes
matching relevant financial documents.

Guidelines:
1. Replace vague pronouns with specific entities from context.
2. Use precise financial terminology: "revenue", "EBITDA", "PAT",
   "ARPU", "OPEX", "margin", "YoY growth", "consolidated", "segment".
3. Break compound questions into a single focused query.
4. Include relevant financial metrics, section names, or time periods
   (e.g., "Q2 FY25", "2Q FY24", "H1 FY25").
5. If conversational/ambiguous, rephrase as concise keyword-style search.
6. Expand ambiguous terms with financial synonyms:
   - "amount spent / cost" → "expenditure", "OPEX", "spending"
   - "consumer base" → "customer base", "subscriber count", "user metrics"
   - "profit" → "EBITDA", "PAT", "net income", "bottom-line"
   - "income" → "revenue", "top-line", "turnover"
   - "operation cost" → "OPEX", "operating expense", "operational cost"
7. Return ONLY the reformulated query string — nothing else."""

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

ANSWER_RELEVANCE_PROMPT = """You are an answer quality evaluator for a
Financial Document Q&A system.

Evaluate whether the SYNTHESIZED ANSWER adequately addresses the USER QUERY
based on the provided DOCUMENT CHUNKS.

## Evaluation Criteria

1. **Direct answer**: The answer contains specific data (numbers, metrics,
   figures, percentages) that directly answer what the user asked.
   → RELEVANT

2. **Related answer with context**: The answer provides related financial
   metrics (e.g., user asked about "cost" but answer has "revenue" or "ARPU")
   AND explicitly explains the connection.
   → RELEVANT

3. **Deflection / no-info response**: The answer says "does not contain",
   "no information found", "not mentioned", or similar — EVEN IF chunks
   contain tangentially related data.
   → NOT RELEVANT (the LLM should have used related data instead)

4. **Generic filler**: The answer restates the question or provides
   generic advice without any specific data from the chunks.
   → NOT RELEVANT

5. **Hallucination**: The answer contains numbers/data NOT present in
   the chunks.
   → NOT RELEVANT

## Input

USER QUERY:
{query}

SYNTHESIZED ANSWER:
{answer}

## Output

Return a JSON object with exactly these keys:
  - "relevant": true or false (boolean)
  - "reason": brief explanation of your judgment (string)

Return ONLY the JSON object. Do NOT wrap it in markdown code fences."""


# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------
class RAGState(TypedDict):
    """State that flows through every node in the graph."""
    messages: Annotated[list, add_messages]
    query: str
    raw_chunks: List[Dict[str, Any]]
    reranked_chunks: List[Dict[str, Any]]
    metadata: List[Dict[str, Any]]
    retries: int
    synthesized_answer: str          # NEW: store answer for relevance check
    answer_relevant: bool            # NEW: relevance evaluation result


# ---------------------------------------------------------------------------
# Helper Utilities
# ---------------------------------------------------------------------------
TOOL_NAMES = {"vector_search", "fts_search", "hybrid_search"}


def _extract_response_text(content: Any) -> str:
    """
    Safely extract a plain string from Gemini's response content.

    Gemini returns content as a list of dicts, e.g.:
        [{"text": "the answer"}]
    but nodes expect a plain string to call .strip(), json.loads(), etc.
    """
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
    """
    Remove markdown code fences (```json ... ``` or ``` ... ```)
    that LLMs sometimes wrap around JSON output.
    """
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.index("\n") if "\n" in text else len(text)
        text = text[first_newline:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        return text.strip()
    return text


def _extract_text_from_chunk(chunk: Dict[str, Any]) -> str:
    """
    Extract searchable text from a chunk, handling:
    - Plain string content
    - JSON string content (tables/images from ingestion.py)
    - List content (from Gemini embeddings with JSONB)
    - Dict content (structured metadata)
    """
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
    """
    Extract and normalize metadata from a chunk.
    Recovers fields from JSON content when metadata is missing.
    """
    if isinstance(chunk, dict):
        meta = chunk.get("metadata", {})
        if isinstance(meta, dict) and "metadata" in meta:
            meta = {**meta, **meta["metadata"]}
    else:
        meta = {}

    # Recover llm_generated_caption / table_data from JSON content
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
    """
    Walk messages backwards and return raw chunk dicts from tool messages.
    """
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


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------
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
        # FIX: On retry, do NOT include old AIMessage/ToolMessage pairs.
        # They cause Gemini's "function call must follow user turn" error.
        # The reformulated query is already in state["query"].
        response = llm_with_tools.invoke([system, user])
    else:
        response = llm_with_tools.invoke([system] + messages + [user])

    logger.info("agent_node (retry=%d): tool_calls=%s", retries, response.tool_calls)
    return {"messages": [response]}

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
                "rerank_node: %d → %d chunks (model=%s)",
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
    """
    Second LLM call — generates the final answer using reranked & scored chunks.
    Stores the raw answer in state for the relevance evaluator.
    """
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

    logger.info("synthesize_node: answer generated (%d chars)", len(answer_text))

    clean_message = AIMessage(content=answer_text)
    return {
        "messages": [clean_message],
        "synthesized_answer": answer_text,
    }


def answer_relevance_node(state: RAGState) -> dict:
    """
    Evaluate whether the synthesized answer actually addresses the user query.
    Uses a dedicated LLM call to judge relevance.

    Returns:
        answer_relevant: True if answer is adequate, False if it should be retried
    """
    query = state["query"]
    answer = state.get("synthesized_answer", "")
    chunks = state.get("reranked_chunks", [])

    if not answer:
        return {"answer_relevant": False}

    # Extract the "answer" field from JSON if present
    answer_text = answer
    try:
        parsed = json.loads(answer)
        if isinstance(parsed, dict) and "answer" in parsed:
            answer_text = parsed["answer"]
    except (json.JSONDecodeError, TypeError):
        pass

    # Build a summary of what chunks were available (for the evaluator context)
    chunks_summary = ""
    if chunks:
        chunk_lines = []
        for i, chunk in enumerate(chunks[:3], start=1):
            text = _extract_text_from_chunk(chunk)
            meta = _get_chunk_metadata(chunk)
            modality = meta.get("modality", "text")
            chunk_lines.append(
                f"Chunk {i} ({modality}, p.{meta.get('page_number', '?')}): "
                f"{text[:300]}..."
            )
        chunks_summary = "\n".join(chunk_lines)

    prompt = ANSWER_RELEVANCE_PROMPT.format(
        query=query,
        answer=answer_text,
    )

    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=chunks_summary if chunks_summary else "No chunks were available."),
    ])

    result_text = _extract_response_text(response.content)
    result_text = _strip_json_fences(result_text)

    # Parse the relevance judgment
    relevant = False
    reason = ""
    try:
        parsed = json.loads(result_text)
        relevant = bool(parsed.get("relevant", False))
        reason = parsed.get("reason", "")
    except (json.JSONDecodeError, TypeError):
        logger.warning("answer_relevance_node: failed to parse LLM output: %s", result_text)
        # If we can't parse, assume relevant to avoid infinite loops
        relevant = True

    logger.info(
        "answer_relevance_node: relevant=%s, reason='%s'",
        relevant, reason[:100],
    )
    return {"answer_relevant": relevant}


def query_reformulate_node(state: RAGState) -> dict:
    """
    When retrieval returns zero relevant chunks OR the answer is not relevant,
    produce a more structured, retrieval-friendly version of the query.
    """
    query = state["query"]
    retries = state.get("retries", 0)
    new_retries = retries + 1

    # Include why we're reformulating — was it no chunks or bad answer?
    answer = state.get("synthesized_answer", "")
    chunks = state.get("reranked_chunks", [])

    reformulate_context = query
    if chunks and answer:
        # We had chunks but the answer wasn't relevant
        answer_text = answer
        try:
            parsed = json.loads(answer)
            if isinstance(parsed, dict) and "answer" in parsed:
                answer_text = parsed["answer"]
        except (json.JSONDecodeError, TypeError):
            pass
        reformulate_context = (
            f"Original query: {query}\n\n"
            f"The previous answer was not relevant: {answer_text[:200]}\n\n"
            f"Reformulate the query to find better documents."
        )
    else:
        reformulate_context = (
            f"Original query: {query}\n\n"
            f"Previous retrieval returned no relevant results.\n"
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
            f"Previous attempt returned {'no relevant answer' if chunks else 'no chunks'}. "
            f"Please use this reformulated query for retrieval:\n"
            f"\"{reformulated}\""
        )
    )

    return {
        "query": reformulated,
        "retries": new_retries,
        "messages": [context_msg],
        "synthesized_answer": "",       # Reset for next iteration
        "answer_relevant": False,        # Reset for next iteration
    }


def extract_chunks_node(state: RAGState) -> dict:
    """Extract raw chunks from the latest ToolMessage."""
    chunks = _extract_chunks_from_messages(state["messages"])
    logger.info("extract_chunks: found %d raw chunks", len(chunks))
    return {"raw_chunks": chunks}


# ---------------------------------------------------------------------------
# Conditional Edge Functions
# ---------------------------------------------------------------------------
def _has_relevant_chunks(state: RAGState) -> str:
    """
    After extract_chunks, decide what to do next.

    Returns:
        "rerank"   — chunks found, proceed to scoring pipeline
        "reformulate" — no chunks and retry budget remains
        "synthesize"  — no chunks and retries exhausted (fallback)
    """
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
    """
    After evaluate_relevance, decide whether to accept or retry.

    Returns:
        "end"         — answer is relevant, finalize
        "reformulate" — answer is not relevant and retry budget remains
    """
    relevant = state.get("answer_relevant", False)
    retries = state.get("retries", 0)

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


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------
def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph RAG pipeline with answer relevance loop.

    Pipeline:
        agent → tools → extract_chunks
            ├── [chunks] → rerank → score → synthesize → evaluate_relevance
            │       ├── [relevant] → END
            │       └── [not relevant + retries < max] → query_reformulate → agent
            │       └── [not relevant + retries exhausted] → END
            └── [no chunks + retries < max] → query_reformulate → agent
            └── [no chunks + retries exhausted] → synthesize (fallback) → END
    """
    graph = StateGraph(RAGState)

    # Nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools=RETRIEVAL_TOOLS))
    graph.add_node("extract_chunks", extract_chunks_node)
    graph.add_node("query_reformulate", query_reformulate_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("score", score_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("evaluate_relevance", answer_relevance_node)

    # Edges
    graph.add_edge(START, "agent")

    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", "__end__": END},
    )

    graph.add_edge("tools", "extract_chunks")

    # After extraction: have chunks? rerank | reformulate | fallback synthesize
    graph.add_conditional_edges(
        "extract_chunks",
        _has_relevant_chunks,
        {
            "rerank": "rerank",
            "reformulate": "query_reformulate",
            "synthesize": "synthesize",
        },
    )

    # Reformulate loops back to agent
    graph.add_edge("query_reformulate", "agent")

    # Normal pipeline: rerank → score → synthesize → evaluate
    graph.add_edge("rerank", "score")
    graph.add_edge("score", "synthesize")
    graph.add_edge("synthesize", "evaluate_relevance")

    # After evaluation: accept answer or reformulate
    graph.add_conditional_edges(
        "evaluate_relevance",
        _check_answer_relevance,
        {
            "end": END,
            "reformulate": "query_reformulate",
        },
    )

    # Fallback synthesize (no chunks, retries exhausted) → END directly
    graph.add_edge("synthesize", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Compiled Graph Singleton
# ---------------------------------------------------------------------------
rag_graph = build_graph()


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def run_rag_agent(query: str, k: int = 10) -> QueryResponse:
    """
    Stateless RAG agent entry point.

    Args:
        query: User's natural language question.
        k: Number of top-k chunks to retrieve (passed to tools).

    Returns:
        QueryResponse with:
          - answer: LLM-synthesized from reranked chunks
          - metadata: programmatic scores + ingestion metadata + provenance
    """
    initial_state: RAGState = {
        "messages": [],
        "query": query,
        "raw_chunks": [],
        "reranked_chunks": [],
        "metadata": [],
        "retries": 0,
        "synthesized_answer": "",
        "answer_relevant": False,
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