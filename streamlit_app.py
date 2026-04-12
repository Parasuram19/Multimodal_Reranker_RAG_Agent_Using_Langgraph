"""
Streamlit GUI for Financial Document RAG System.

Pages:
  💬 Chat    — Query the RAG agent via API
  📤 Upload  — Admin: ingest PDF documents via API

Run:
  streamlit run app.py --server.port 8501
"""

import json
import requests
import streamlit as st
from streamlit_cookies_controller import CookieController
from dotenv import load_dotenv
load_dotenv(override=True)
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial RAG Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = "http://127.0.0.1:8000"
QUERY_ENDPOINT = f"{API_BASE_URL}/api/v1/query"
UPLOAD_ENDPOINT = f"{API_BASE_URL}/api/v1/admin/upload"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "upload_status" not in st.session_state:
    st.session_state.upload_status = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def check_api_health() -> bool:
    """Check if the backend API is reachable."""
    try:
        resp = requests.get(HEALTH_ENDPOINT, timeout=5)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def query_rag(query: str, k: int = 10) -> dict:
    """Call the RAG query endpoint."""
    payload = {"query": query, "k": k}
    try:
        resp = requests.post(
            QUERY_ENDPOINT,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The query may be too complex."}
    except requests.exceptions.ConnectionError:
        return {"error": f"Cannot connect to API at {API_BASE_URL}. Is the server running?"}
    except requests.exceptions.HTTPError as e:
        return {"error": f"API error: {e.response.status_code} — {e.response.text[:300]}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


def upload_document(file, doc_name: str, keywords: list) -> dict:
    """Call the ingestion/upload endpoint."""
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        data = {
            "doc_name": doc_name,
            "keywords": json.dumps(keywords),
        }
        resp = requests.post(
            UPLOAD_ENDPOINT,
            files=files,
            data=data,
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        return {"error": "Upload timed out. The document may be too large."}
    except requests.exceptions.ConnectionError:
        return {"error": f"Cannot connect to API at {API_BASE_URL}. Is the server running?"}
    except requests.exceptions.HTTPError as e:
        return {"error": f"API error: {e.response.status_code} — {e.response.text[:300]}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


def render_metadata_table(metadata: list):
    """Render retrieved chunk metadata as an expandable table."""
    if not metadata:
        return

    with st.expander("📎 Retrieved Chunks & Scores", expanded=False):
        for i, meta in enumerate(metadata, start=1):
            rank = meta.get("rank", i)
            score = meta.get("relevance_score", 0)
            rerank = meta.get("rerank_score")
            doc = meta.get("document_name", "unknown")
            page = meta.get("page_number", "?")
            modality = meta.get("modality", "text")
            section = meta.get("section_header", "")

            header = f"**#{rank}** — {doc} (p.{page}) | {modality} | Score: {score:.4f}"
            if rerank is not None:
                header += f" | Rerank: {rerank:.4f}"

            citation = meta.get("content", "")[:200]
            if len(meta.get("content", "")) > 200:
                citation += "..."

            col1, col2 = st.columns([1, 4])
            with col1:
                st.caption(f"**Section:** {section}")
            with col2:
                st.caption(header)
                st.text(citation)
            st.divider()


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Chat bubbles */
    .user-msg {
        background: linear-gradient(135deg, #1e3a5f, #2563eb);
        color: white;
        padding: 12px 16px;
        border-radius: 16px 16px 4px 16px;
        margin: 8px 0;
        max-width: 75%;
        margin-left: auto;
        font-size: 14px;
    }
    .assistant-msg {
        background: #f1f5f9;
        color: #1e293b;
        padding: 12px 16px;
        border-radius: 16px 16px 16px 4px;
        margin: 8px 0;
        max-width: 85%;
        font-size: 14px;
        border: 1px solid #e2e8f0;
    }
    .error-msg {
        background: #fef2f2;
        color: #991b1b;
        padding: 12px 16px;
        border-radius: 16px 16px 16px 4px;
        margin: 8px 0;
        border: 1px solid #fecaca;
        font-size: 14px;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📊 Financial RAG")
    st.caption("Insurance & Policy Document Q&A")

    st.divider()

    # API status
    api_ok = check_api_health()
    if api_ok:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Unreachable")
        st.caption(f"Endpoint: `{API_BASE_URL}`")

    st.divider()

    # Settings
    st.subheader("⚙️ Settings")
    st.text_input("API Base URL", value=API_BASE_URL, key="api_url_input",
                  disabled=True, help="Set via .streamlit/secrets.toml")

    top_k = st.slider("Top-K Chunks", min_value=1, max_value=20, value=10, step=1)

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear Chat History", use_container_width=True, type="secondary"):
        st.session_state.chat_history = []
        st.rerun()


# ---------------------------------------------------------------------------
# Page Router
# ---------------------------------------------------------------------------
page = st.radio(
    "Navigation",
    ["💬 Chat", "📤 Upload (Admin)"],
    horizontal=True,
    label_visibility="collapsed",
)


# ===========================================================================
# PAGE 1: Chat
# ===========================================================================
if page == "💬 Chat":

    # Header
    st.subheader("💬 Ask about Financial Documents")
    st.caption("Query your ingested financial documents — tables, charts, and text.")

    # Display chat history
    for turn in st.session_state.chat_history:
        role = turn["role"]
        content = turn["content"]
        metadata = turn.get("metadata")

        if role == "user":
            st.markdown(f'<div class="user-msg">{content}</div>', unsafe_allow_html=True)
        elif role == "assistant":
            if "error" in turn:
                st.markdown(f'<div class="error-msg">⚠️ {turn["error"]}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-msg">{content}</div>',
                            unsafe_allow_html=True)
                if metadata:
                    render_metadata_table(metadata)

    # Chat input
    if prompt := st.chat_input("Ask a question about your financial documents...",
                               key="chat_input"):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Display user bubble immediately
        st.markdown(f'<div class="user-msg">{prompt}</div>', unsafe_allow_html=True)

        # Show loading
        with st.spinner("🔍 Retrieving & synthesizing answer..."):
            result = query_rag(prompt, k=top_k)

        if "error" in result:
            st.markdown(f'<div class="error-msg">⚠️ {result["error"]}</div>',
                        unsafe_allow_html=True)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "",
                "error": result["error"],
            })
        else:
            answer = result.get("answer", "No answer returned.")
            metadata = result.get("metadata", [])

            st.markdown(f'<div class="assistant-msg">{answer}</div>',
                        unsafe_allow_html=True)

            if metadata:
                render_metadata_table(metadata)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "metadata": metadata,
            })


# ===========================================================================
# PAGE 2: Upload (Admin)
# ===========================================================================
elif page == "📤 Upload (Admin)":

    st.subheader("📤 Ingest Financial Documents")
    st.caption("Upload PDF documents to be indexed for retrieval.")

    st.divider()

    # Simple password gate (optional)
    with st.expander("🔒 Admin Access", expanded=True):
        admin_key = st.text_input("Admin Key", type="password",
                                  placeholder="Enter admin key to enable upload")
        admin_authenticated = (
            admin_key == st.secrets.get("ADMIN_KEY", "admin123")
        )

    if not admin_authenticated and admin_key:
        st.error("❌ Invalid admin key")
        st.stop()
    elif not admin_key:
        st.info("Enter admin key to enable the upload form.")
        st.stop()

    # Upload form
    st.divider()
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            label_visibility="visible",
            help="Upload a financial document (PDF) for ingestion.",
        )

    with col2:
        doc_name = st.text_input(
            "Document Name",
            placeholder="e.g., RIL-Q2-FY25-Results",
            help="A human-readable name for this document.",
        )

    st.divider()

    # Keywords
    keywords_input = st.text_area(
        "Search Keywords (comma-separated)",
        placeholder="e.g., Revenue, EBITDA, Jio, Reliance, Q2 FY25",
        help="Keywords help with retrieval. Separate with commas.",
        height=80,
    )

    # Upload button
    if uploaded_file and doc_name:
        st.divider()
        if st.button("🚀 Ingest Document", type="primary", use_container_width=True):
            keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]

            with st.status("Ingesting document...", expanded=True) as status:
                st.write(f"📄 File: `{uploaded_file.name}`")
                st.write(f"📝 Name: `{doc_name}`")
                st.write(f"🏷️ Keywords: {keywords}")

                st.write("---")
                st.write("⏳ Parsing PDF with Docling...")

                result = upload_document(uploaded_file, doc_name, keywords)

                if "error" in result:
                    st.error(f"❌ {result['error']}")
                    status.update(label="Ingestion Failed", state="error")
                else:
                    st.write(f"✅ Document ID: `{result.get('document_id', 'N/A')}`")
                    st.write(f"📊 Chunks indexed: {result.get('indexed_chunks', 'N/A')}")
                    st.write(f"📄 Total pages: {result.get('total_pages', 'N/A')}")
                    status.update(label="Ingestion Complete", state="complete")
    elif uploaded_file:
        st.warning("Please enter a document name before uploading.")