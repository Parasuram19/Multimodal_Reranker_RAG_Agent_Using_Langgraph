"""
NorthStar Bank — Agentic RAG Assistant (Streamlit Frontend)

Connects to the FastAPI backend at http://127.0.0.1:8000/api/v1
  - POST /admin/upload  -> upload & ingest a PDF into the knowledge base
  - POST /query          -> run the RAG agent and get answer + metadata

Every assistant response shows an expandable "Citations & Chunks" panel
below it with full JSON metadata for each retrieved chunk / SQL result.
"""

import json
import time
from datetime import datetime

import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────
API_BASE = "http://127.0.0.1:8000/api/v1"

st.set_page_config(
    page_title="NorthStar Bank RAG Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    /* Hide default Streamlit hamburger & footer for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Citation cards */
    .citation-card {
        border-left: 4px solid #4CAF50;
        padding: 10px 14px;
        margin: 6px 0;
        background: #f8f9fa;
        border-radius: 4px;
        font-size: 13px;
    }
    .citation-card.sql {
        border-left-color: #2196F3;
    }
    .citation-card.doc {
        border-left-color: #4CAF50;
    }
    .citation-card img-card {
        border-left-color: #FF9800;
    }
    .citation-card table-card {
        border-left-color: #9C27B0;
    }

    /* Route badges */
    .route-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        color: white;
    }
    .route-badge.sql { background: #2196F3; }
    .route-badge.document { background: #4CAF50; }
    .route-badge.hybrid { background: #FF9800; }

    /* Metadata table */
    .meta-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }
    .meta-table th {
        background: #e8f5e9;
        text-align: left;
        padding: 6px 10px;
        border: 1px solid #ddd;
    }
    .meta-table td {
        padding: 6px 10px;
        border: 1px solid #eee;
    }

    /* Chat message spacing */
    .stChatMessage { padding-bottom: 0 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"streamlit_{int(time.time())}"


# ─────────────────────────────────────────────────────────────────────────
# Citation Renderer
# ─────────────────────────────────────────────────────────────────────────
def _render_citations(metadata: list, msg_idx: int):
    """
    Render an expandable panel showing all citations/chunks for a response.

    Each citation shows:
    - Rank, modality badge, document name, page, section
    - Content preview
    - Full JSON metadata (in expandable sub-section)
    """
    if not metadata:
        return

    expander_label = f"Citations & Chunks ({len(metadata)} result{'s' if len(metadata) != 1 else ''})"
    with st.expander(expander_label, expanded=False):
        for i, meta in enumerate(metadata):
            rank = meta.get("rank", i + 1)
            modality = meta.get("modality", "text")
            route = meta.get("route", "")
            doc_name = meta.get("document_name", "Unknown")
            page = meta.get("page_number", meta.get("page", "N/A"))
            section = meta.get("section_header", "")
            content = meta.get("content", "")
            citation_text = meta.get("citation", "")

            # Determine card class based on modality and route
            card_class = "citation-card doc"
            if route == "sql":
                card_class = "citation-card sql"
            elif modality == "image":
                card_class = "citation-card img-card"
            elif modality == "table":
                card_class = "citation-card table-card"

            # Modality icon
            modality_icons = {
                "text": "📄",
                "table": "📊",
                "image": "🖼️",
            }
            mod_icon = modality_icons.get(modality, "📄")

            # Route icon
            route_icons = {
                "sql": "🗄️",
                "document": "📑",
                "hybrid": "🔀",
            }
            route_icon = route_icons.get(route, "")

            # Build citation card
            card_html = f"""
            <div class="{card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <strong>#{rank} {mod_icon} {modality.upper()}</strong>
                    <span style="font-size: 11px; color: #888;">
                        {doc_name} | Page {page}
                    </span>
                </div>
            """
            if section:
                card_html += f'<div style="font-size: 12px; color: #555; margin-bottom: 4px;">📂 {section}</div>'

            if route == "sql" and meta.get("sql_query"):
                card_html += f'<div style="font-size: 12px; color: #2196F3; margin-bottom: 4px;">🗄️ SQL: <code>{meta["sql_query"][:200]}</code></div>'
                if meta.get("sql_row_count") is not None:
                    card_html += f'<div style="font-size: 11px; color: #888;">Rows returned: {meta["sql_row_count"]}</div>'

            if content:
                preview = content[:300] + "..." if len(content) > 300 else content
                card_html += f'<div style="font-size: 12px; margin-top: 6px; line-height: 1.5;">{preview}</div>'

            card_html += "</div>"
            st.markdown(card_html, unsafe_allow_html=True)

            # Scores (if available)
            scores = {}
            for score_key in [
                "cosine_similarity",
                "bm25",
                "relevance_score",
                "rerank_score",
            ]:
                val = meta.get(score_key)
                if val is not None:
                    scores[score_key] = round(float(val), 4) if isinstance(val, (int, float)) else val

            if scores:
                cols = st.columns(len(scores))
                for j, (skey, sval) in enumerate(scores.items()):
                    with cols[j]:
                        label = skey.replace("_", " ").title()
                        color = "green" if skey == "relevance_score" and sval > 0.5 else "normal"
                        st.metric(label=label, value=sval)

            # Full JSON expandable
            with st.expander("View Raw JSON", expanded=False):
                # Clean up metadata for display (remove huge fields)
                display_meta = {
                    k: v for k, v in meta.items()
                    if k != "image_base64"  # skip huge base64 images
                }
                st.json(display_meta)

            if i < len(metadata) - 1:
                st.divider()
                
# ─────────────────────────────────────────────────────────────────────────
# Sidebar — Document Upload
# ─────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Document Management")
    st.markdown("---")

    st.subheader("Upload PDF")
    uploaded_file = st.file_uploader(
        "Select a PDF to ingest into the Knowledge Base",
        type=["pdf"],
        key="pdf_uploader",
        help="Only PDF files are accepted. The document will be parsed, chunked, and embedded.",
    )

    if uploaded_file is not None:
        file_details = {
            "Filename": uploaded_file.name,
            "Type": uploaded_file.type,
            "Size": f"{uploaded_file.size / 1024:.1f} KB",
        }
        with st.expander("File Details", expanded=False):
            st.json(file_details)

        if st.button("Upload & Ingest", type="primary", use_container_width=True):
            with st.status("Uploading and processing document...", expanded=True) as status:
                st.write("Sending file to ingestion pipeline...")
                try:
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    }
                    response = requests.post(
                        f"{API_BASE}/admin/upload",
                        files=files,
                        timeout=300,  # ingestion can take a while for large PDFs
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.write("Ingestion completed successfully!")
                        st.success(
                            f"File `{result.get('file', uploaded_file.name)}` "
                            f"uploaded and ingested."
                        )
                        status.update(
                            label="Ingestion Complete",
                            state="complete",
                        )
                    else:
                        error_detail = response.text
                        try:
                            error_detail = response.json().get("detail", response.text)
                        except Exception:
                            pass
                        st.error(f"Upload failed (HTTP {response.status_code}): {error_detail}")
                        status.update(label="Ingestion Failed", state="error")

                except requests.exceptions.ConnectionError:
                    st.error(
                        "Cannot connect to the backend API. "
                        f"Make sure the server is running at {API_BASE}"
                    )
                    status.update(label="Connection Error", state="error")
                except requests.exceptions.Timeout:
                    st.error(
                        "Upload timed out. The document may be too large "
                        "or the ingestion pipeline is still processing."
                    )
                    status.update(label="Timeout", state="error")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")
                    status.update(label="Error", state="error")

    st.markdown("---")

    # Session controls
    st.subheader("Session")
    st.caption(f"Thread ID: `{st.session_state.thread_id[:16]}...`")

    if st.button("New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = f"streamlit_{int(time.time())}"
        st.rerun()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.subheader("About")
    st.markdown(
        """
        **NorthStar Bank RAG Assistant**

        This assistant can answer questions about:
        - Banking products & services
        - Account & transaction data
        - Loan & deposit details
        - Credit card information
        - Product terms & policies

        **Routes:**
        - **SQL** — structured banking data
        - **Document** — knowledge base PDFs
        - **Hybrid** — both sources combined
        """
    )

# ─────────────────────────────────────────────────────────────────────────
# Main Chat Area
# ─────────────────────────────────────────────────────────────────────────
st.title("NorthStar Bank RAG Assistant")
st.caption(
    "Ask questions about banking products, account data, policies, and more. "
    "Upload PDFs to expand the knowledge base."
)

# ── Render existing messages ─────────────────────────────────────────────
for idx, message in enumerate(st.session_state.messages):
    role = message["role"]

    with st.chat_message(role):
        # Main answer
        st.markdown(message["content"])

        # Citations & Chunks panel (only for assistant messages with metadata)
        if role == "assistant" and message.get("metadata"):
            _render_citations(message["metadata"], idx)


# ── Chat input ───────────────────────────────────────────────────────────
if prompt := st.chat_input(
    "Ask about NorthStar Bank products, accounts, loans, deposits, credit cards..."
):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt, "metadata": None})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_placeholder.markdown("Thinking...")

        try:
            start_time = time.time()
            response = requests.post(
                f"{API_BASE}/query",
                json={"query": prompt},
                timeout=120,
            )
            elapsed = time.time() - start_time

            if response.status_code == 200:
                data = response.json()

                answer = data.get("answer", "No answer received.")
                metadata = data.get("metadata", [])

                # Determine route for badge
                route = "document"
                for m in metadata:
                    if m.get("route"):
                        route = m["route"]
                        break

                # Build rich answer with route badge
                route_badge_class = f"route-badge {route}"
                route_label = route.upper()
                rich_answer = (
                    f'<span class="route-badge {route_badge_class}">{route_label}</span>\n\n'
                    f'{answer}'
                )
                response_placeholder.markdown(rich_answer, unsafe_allow_html=True)

                # Show timing
                st.caption(f"Response time: {elapsed:.1f}s | Route: {route}")

                # Render citations
                if metadata:
                    _render_citations(metadata, len(st.session_state.messages) - 1)

                # Save to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": rich_answer,
                    "metadata": metadata,
                    "raw_answer": answer,
                    "route": route,
                })

            else:
                error_detail = response.text
                try:
                    error_detail = response.json().get("detail", response.text)
                except Exception:
                    pass
                response_placeholder.error(
                    f"Error (HTTP {response.status_code}): {error_detail}"
                )

        except requests.exceptions.ConnectionError:
            response_placeholder.error(
                f"Cannot connect to the backend at `{API_BASE}`. "
                "Please ensure the FastAPI server is running."
            )
        except requests.exceptions.Timeout:
            response_placeholder.error(
                "Request timed out. The query may be too complex or the server is busy."
            )
        except Exception as e:
            response_placeholder.error(f"Unexpected error: {str(e)}")


