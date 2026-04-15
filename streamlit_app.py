#!/usr/bin/env python3
"""
NorthStar Bank - Smart RAG Assistant
Streamlit frontend with complete metadata display for retrieved chunks.
"""

import streamlit as st
import requests
import json
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ============================================================================
# Config
# ============================================================================
BASE_URL = "http://127.0.0.1:8000/api/v1"
UPLOAD_ENDPOINT = f"{BASE_URL}/admin/upload"
QUERY_ENDPOINT = f"{BASE_URL}/query"

st.set_page_config(
    page_title="NorthStar Smart RAG 🏦",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Custom CSS
# ============================================================================
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1200px; }
    html, body, [class*="st-"] { font-family: 'Segoe UI', system-ui, sans-serif; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    
    /* Chat bubbles */
    .stChatMessage { border-radius: 12px; padding: 1rem; }
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background: #f8fafc; border: 1px solid #e2e8f0;
    }
    .stChatMessage[data-testid="stChatMessageUser"] {
        background: #eff6ff; border: 1px solid #bfdbfe;
    }
    
    /* Citation pills */
    .citation-pill {
        display: inline-flex; align-items: center; gap: 4px;
        padding: 3px 10px; margin: 2px 4px 2px 0;
        border-radius: 9999px; font-size: 0.75rem; font-weight: 500;
        border: 1px solid #cbd5e1; background: #f1f5f9; color: #334155;
        cursor: pointer; transition: all 0.15s ease;
    }
    .citation-pill:hover { background: #e2e8f0; border-color: #94a3b8; }
    .citation-pill .badge {
        background: #3b82f6; color: white; border-radius: 9999px;
        padding: 1px 6px; font-size: 0.65rem; margin-right: 2px;
    }
    
    /* Route badges */
    .route-badge {
        display: inline-flex; align-items: center; gap: 4px;
        padding: 2px 10px; border-radius: 6px; font-size: 0.7rem;
        font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .route-document { background: #dbeafe; color: #1d4ed8; border: 1px solid #93c5fd; }
    .route-sql     { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
    .route-hybrid  { background: #fef3c7; color: #b45309; border: 1px solid #fde68a; }
    
    /* Chunk cards - complete metadata display */
    .chunk-card {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 8px; padding: 12px 16px; margin: 8px 0;
        font-size: 0.85rem; line-height: 1.5; color: #334155;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .chunk-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    
    .chunk-meta {
        display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px;
        font-size: 0.7rem; color: #64748b; font-weight: 500;
    }
    .chunk-meta span {
        background: #f1f5f9; padding: 2px 8px; border-radius: 4px;
        border: 1px solid #e2e8f0;
    }
    .chunk-meta .key { color: #475569; font-weight: 600; }
    .chunk-meta .value { color: #334155; }
    
    .chunk-content {
        white-space: pre-wrap; word-break: break-word;
        background: #f8fafc; padding: 10px; border-radius: 6px;
        border-left: 3px solid #3b82f6; margin: 8px 0;
    }
    .chunk-content img {
        max-width: 100%; border-radius: 6px; margin: 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Metadata table */
    .meta-table {
        width: 100%; border-collapse: collapse; font-size: 0.75rem;
        margin: 8px 0; background: #f8fafc; border-radius: 6px;
        overflow: hidden;
    }
    .meta-table th {
        text-align: left; padding: 6px 10px; background: #e2e8f0;
        font-weight: 600; color: #475569; border-bottom: 1px solid #cbd5e1;
    }
    .meta-table td {
        padding: 6px 10px; border-bottom: 1px solid #f1f5f9;
        color: #334155;
    }
    .meta-table tr:last-child td { border-bottom: none; }
    
    /* Upload zone */
    .upload-container {
        border: 2px dashed #cbd5e1; border-radius: 16px;
        padding: 3rem 2rem; text-align: center; background: #f8fafc;
        transition: all 0.2s ease;
    }
    .upload-container:hover { border-color: #3b82f6; background: #eff6ff; }
    
    /* Status cards */
    .status-card { padding: 16px; border-radius: 10px; border: 1px solid; }
    .status-success { background: #f0fdf4; border-color: #86efac; color: #15803d; }
    .status-error { background: #fef2f2; border-color: #fca5a5; color: #dc2626; }
    .status-info { background: #eff6ff; border-color: #93c5fd; color: #1d4ed8; }
    
    /* Hide Streamlit chrome */
    #MainMenu, footer, .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Session State
# ============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "active_page" not in st.session_state:
    st.session_state.active_page = "chat"

# ============================================================================
# API Helpers
# ============================================================================
def upload_document(file_obj, file_name: str) -> Dict[str, Any]:
    """Upload PDF to ingestion endpoint."""
    try:
        files = {"file": (file_name, file_obj, "application/pdf")}
        response = requests.post(UPLOAD_ENDPOINT, files=files, timeout=300)
        return {
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "data": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
        }
    except requests.exceptions.ConnectionError:
        return {"success": False, "status_code": 0, "data": f"Cannot connect to {BASE_URL}"}
    except requests.exceptions.Timeout:
        return {"success": False, "status_code": 0, "data": "Upload timed out"}
    except Exception as e:
        return {"success": False, "status_code": 0, "data": str(e)}


def send_query(query: str, session_id: str) -> Dict[str, Any]:
    """Send query to RAG endpoint."""
    try:
        payload = {"query": query, "session_id": session_id}
        response = requests.post(QUERY_ENDPOINT, json=payload, timeout=120)
        if response.status_code == 200:
            data = response.json()
            return normalize_response(data)
        return {
            "answer": f"Error: Server returned {response.status_code}",
            "route": None, "citations": [], "retrieved_chunks": [], "raw": response.text,
        }
    except requests.exceptions.ConnectionError:
        return {"answer": "Cannot connect to server", "route": None, "citations": [], "retrieved_chunks": [], "raw": None}
    except Exception as e:
        return {"answer": f"Request failed: {e}", "route": None, "citations": [], "retrieved_chunks": [], "raw": None}


def normalize_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize API response to standard format."""
    result = {
        "answer": "", "route": None, "citations": [], "retrieved_chunks": [], "raw": data,
    }
    
    # Extract answer
    for key in ("answer", "response", "result", "output", "message", "text"):
        if key in data and data[key]:
            result["answer"] = data[key]
            break
    if not result["answer"]:
        for wrapper in ("data", "output"):
            if wrapper in data and isinstance(data[wrapper], dict):
                for key in ("answer", "response", "result"):
                    if key in data[wrapper] and data[wrapper][key]:
                        result["answer"] = data[wrapper][key]
                        break
    
    # Extract route
    for key in ("route", "routing_decision", "path", "strategy", "source"):
        if key in data:
            result["route"] = str(data[key]).lower()
            break
    
    # Extract citations/metadata
    for key in ("citations", "sources", "metadata", "references"):
        if key in data:
            items = data[key]
            if isinstance(items, list):
                result["citations"] = items
            elif isinstance(items, dict):
                result["citations"] = [items]
            break
    
    # Extract retrieved chunks
    for key in ("retrieved_chunks", "chunks", "documents", "context", "raw_chunks"):
        if key in data:
            items = data[key]
            if isinstance(items, list):
                result["retrieved_chunks"] = items
            break
        # Check nested
        if "data" in data and isinstance(data["data"], dict):
            if key in data["data"] and isinstance(data["data"][key], list):
                result["retrieved_chunks"] = data["data"][key]
                break
    
    return result

# ============================================================================
# UI Components
# ============================================================================
def render_sidebar():
    """Render sidebar navigation."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0 2rem 0;">
            <div style="font-size: 2.5rem;">🏦</div>
            <div style="font-size: 1.1rem; font-weight: 700;">NorthStar Bank</div>
            <div style="font-size: 0.75rem; color: #94a3b8;">Smart RAG Assistant</div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()
        
        # Navigation
        nav_cols = st.columns(2)
        with nav_cols[0]:
            if st.button("💬 Chat", use_container_width=True, 
                        type="primary" if st.session_state.active_page == "chat" else "secondary"):
                st.session_state.active_page = "chat"
                st.rerun()
        with nav_cols[1]:
            if st.button("📤 Upload", use_container_width=True,
                        type="primary" if st.session_state.active_page == "upload" else "secondary"):
                st.session_state.active_page = "upload"
                st.rerun()
        
        st.divider()
        
        # Session info
        st.markdown("#### Session")
        st.caption(f"ID: `{st.session_state.session_id}`")
        if st.button("🔄 New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())[:8]
            st.rerun()
        
        st.divider()
        
        # Stats
        if st.session_state.messages:
            st.markdown("#### Stats")
            user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
            total_cites = sum(len(m.get("citations", [])) for m in st.session_state.messages if m["role"] == "assistant")
            st.metric("Questions", len(user_msgs))
            st.metric("Citations", total_cites)
        
        st.divider()
        
        # API Status
        st.markdown("#### API Status")
        try:
            r = requests.get(f"{BASE_URL.replace('/api/v1', '')}/health", timeout=3)
            status = "● Connected" if r.ok else "● Degraded"
            color = "#4ade80" if r.ok else "#facc15"
            st.markdown(f'<span style="color: {color};">{status}</span>', unsafe_allow_html=True)
        except:
            st.markdown('<span style="color: #f87171;">● Disconnected</span>', unsafe_allow_html=True)
        st.caption(f"`{BASE_URL}`")


def render_citations(citations: List[Any]):
    """Render citation pills with deduplication."""
    if not citations:
        return
    
    st.markdown('<div style="margin: 10px 0 4px 0; font-size: 0.78rem; color: #64748b; font-weight: 600;">📋 Sources</div>', unsafe_allow_html=True)
    
    pills_html = ""
    seen = set()
    
    for cit in citations:
        if isinstance(cit, dict):
            doc = cit.get("source_file") or cit.get("document_name") or cit.get("source") or cit.get("source_filename") or ""
            page = cit.get("page") or cit.get("page_no") or cit.get("page_number") or ""
            section = cit.get("section") or cit.get("section_header") or ""
            modality = cit.get("modality") or cit.get("element_type") or ""
            
            dedup_key = f"{doc}:{page}:{section}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            
            # Modality icon
            mod_icon = {"text": "📄", "table": "📊", "image": "🖼️", "mixed": "📎"}.get(modality, "📄")
            
            # Build pill text
            parts = [Path(doc).stem if doc else "Unknown"]
            if page and str(page) not in ("0", "None"):
                parts.append(f"p.{page}")
            if section:
                parts.append(section[:25])
            
            pill_text = " · ".join(parts)
            pills_html += f'<span class="citation-pill"><span class="badge">{mod_icon}</span>{pill_text}</span>'
        elif isinstance(cit, str):
            if cit not in seen:
                seen.add(cit)
                pills_html += f'<span class="citation-pill">📄 {cit[:40]}...</span>'
    
    if pills_html:
        st.markdown(f'<div style="display: flex; flex-wrap: wrap; gap: 2px;">{pills_html}</div>', unsafe_allow_html=True)


def _format_metadata_value(value: Any) -> str:
    """Format metadata value for display."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "✓" if value else "✗"
    if isinstance(value, (int, float)):
        return f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value[:3]) + ("..." if len(value) > 3 else "")
    if isinstance(value, dict):
        return json.dumps(value, indent=2)[:100] + "..." if len(json.dumps(value)) > 100 else json.dumps(value)
    return str(value)


def render_chunk_metadata(meta: Dict[str, Any]):
    """Render complete chunk metadata in a formatted table."""
    if not meta:
        return
    
    # Organize metadata into logical groups
    groups = {
        "🆔 Identity": ["chunk_index", "chunk_level", "is_parent", "parent_chunk_id", "child_chunk_ids", "content_hash"],
        "📄 Document": ["document_name", "source_filename", "document_id", "ingested_at", "version"],
        "📍 Location": ["section", "page", "page_start", "page_end", "header_metadata"],
        "🏷️  Type": ["modality", "element_type", "product_category", "region"],
        "📊 Scores": ["cosine_similarity", "bm25", "relevance_score", "rerank_score", "rerank_index", "rank"],
        "🖼️  Media": ["image_count", "table_count", "docling_caption", "table_data"],
        "🔗 Links": ["search_keywords", "retrieval_weight"],
    }
    
    html_parts = ['<table class="meta-table">']
    
    for group_name, keys in groups.items():
        rows = []
        for key in keys:
            if key in meta and meta[key] is not None:
                # Skip large fields in summary table
                if key in ("image_base64", "section_images", "table_data") and len(str(meta[key])) > 200:
                    continue
                val = _format_metadata_value(meta[key])
                rows.append(f'<tr><td class="key">{key}</td><td class="value">{val}</td></tr>')
        
        if rows:
            html_parts.append(f'<tr><th colspan="2" style="background:#cbd5e1;">{group_name}</th></tr>')
            html_parts.extend(rows)
    
    html_parts.append('</table>')
    st.markdown("".join(html_parts), unsafe_allow_html=True)
    
    # Expandable full metadata JSON
    with st.expander("🔍 View Full Metadata JSON", expanded=False):
        st.json(meta, expanded=False)


def render_chunk_content(chunk: Dict[str, Any], chunk_idx: int):
    """Render chunk content with images, tables, and complete metadata."""
    if not isinstance(chunk, dict):
        st.markdown(f'<div class="chunk-card"><div class="chunk-content">{str(chunk)[:2000]}</div></div>', unsafe_allow_html=True)
        return
    
    # Extract content
    content = (
        chunk.get("content") or chunk.get("text") or chunk.get("page_content") 
        or chunk.get("body") or str(chunk.get("metadata", {}))
    )
    meta = chunk.get("metadata", {})
    
    # Handle metadata stored as JSON string
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except:
            meta = {}
    
    # Chunk header with key metadata
    doc_name = meta.get("source_filename") or meta.get("document_name") or "Unknown"
    page = meta.get("page") or meta.get("page_no") or ""
    section = meta.get("section") or ""
    modality = meta.get("modality") or meta.get("element_type") or "text"
    chunk_level = meta.get("chunk_level", "")
    
    header_parts = []
    header_parts.append(f"📄 {Path(doc_name).stem}")
    if page and str(page) not in ("0", "None"):
        header_parts.append(f"📑 p.{page}")
    if section:
        header_parts.append(f"📌 {section[:35]}")
    mod_icon = {"text": "📝", "table": "📊", "image": "🖼️", "mixed": "📎"}.get(modality, "📄")
    header_parts.append(f"{mod_icon} {modality}")
    if chunk_level:
        header_parts.append(f"{'👨‍👦' if chunk_level == 'parent' else '👶'} {chunk_level}")
    
    # Scores if available
    if meta.get("relevance_score"):
        header_parts.append(f"🎯 {meta['relevance_score']:.3f}")
    if meta.get("rerank_score"):
        header_parts.append(f"⭐ {meta['rerank_score']:.3f}")
    
    meta_html = "".join(f'<span><span class="key">{p.split()[0]}</span> {p}</span>' if " " in p else f'<span>{p}</span>' for p in header_parts)
    
    # Content display
    display_content = content[:3000]
    if len(content) > 3000:
        display_content += f"\n\n... [truncated, full: {len(content)} chars]"
    
    # Image rendering
    image_html = ""
    if meta.get("image_base64"):
        mime = meta.get("image_mime_type", "image/jpeg")
        caption = meta.get("docling_caption", "Embedded image")
        width = meta.get("image_width", 400)
        height = meta.get("image_height", 300)
        image_html = f'''
        <div style="margin: 10px 0; text-align: center;">
            <img src="data:{mime};base64,{meta['image_base64']}" 
                 alt="{caption}" 
                 style="max-width:100%; max-height:400px; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.1);"
                 loading="lazy">
            <div style="font-size:0.75rem; color:#64748b; margin-top:4px;">🖼️ {caption} ({width}×{height})</div>
        </div>'''
    
    # Table rendering
    table_html = ""
    if meta.get("table_data"):
        table_content = meta["table_data"][:2000]
        table_html = f'''
        <div style="background:#f8fafc; padding:10px; border-radius:6px; margin:8px 0; 
                    border-left:3px solid #8b5cf6; overflow-x:auto; font-size:0.8rem;">
            <div style="font-weight:600; color:#475569; margin-bottom:6px;">📊 Table Data:</div>
            <pre style="white-space:pre-wrap; word-break:break-word; margin:0;">{table_content}</pre>
        </div>'''
    
    # Section images list (if parent has multiple images)
    section_images_html = ""
    if meta.get("section_images"):
        imgs = meta["section_images"]
        if isinstance(imgs, list) and imgs:
            section_images_html = f'''
            <div style="background:#f0f9ff; padding:10px; border-radius:6px; margin:8px 0; 
                        border-left:3px solid #0ea5e9;">
                <div style="font-weight:600; color:#0369a1; margin-bottom:6px;">
                    🖼️ Section contains {len(imgs)} image(s):
                </div>
                <ul style="margin:0; padding-left:20px; font-size:0.8rem;">
            '''
            for img in imgs[:3]:  # Show first 3
                cap = img.get("docling_caption", "Image")[:50]
                page_num = img.get("page", "?")
                section_images_html += f'<li>📑 p.{page_num}: {cap}</li>'
            if len(imgs) > 3:
                section_images_html += f'<li>... and {len(imgs) - 3} more</li>'
            section_images_html += '</ul></div>'
    
    # Render the chunk card
    st.markdown(
        f'''<div class="chunk-card">
            <div class="chunk-meta">{meta_html}</div>
            <div class="chunk-content">{display_content}</div>
            {image_html}
            {table_html}
            {section_images_html}
        </div>''',
        unsafe_allow_html=True,
    )
    
    # Render complete metadata table
    render_chunk_metadata(meta)
    
    # Parent-Child relationship indicator
    if meta.get("is_parent") and meta.get("child_chunk_ids"):
        child_count = len(meta["child_chunk_ids"])
        st.caption(f"👨‍👦 Parent chunk with {child_count} child chunk(s) for retrieval")
    elif not meta.get("is_parent") and meta.get("parent_chunk_id"):
        st.caption(f"👶 Child chunk → Parent: `{meta['parent_chunk_id']}`")


def render_retrieved_chunks(chunks: List[Any], msg_index: int):
    """Render all retrieved chunks with complete metadata display."""
    if not chunks:
        return
    
    with st.expander(f"🔍 Retrieved Chunks ({len(chunks)})", expanded=False):
        for i, chunk in enumerate(chunks):
            st.markdown(f"#### Chunk {i+1}")
            render_chunk_content(chunk, i)
            if i < len(chunks) - 1:
                st.divider()


def render_route_badge(route: Optional[str]):
    """Render routing decision badge."""
    if not route:
        return
    
    route_str = str(route).lower()
    if "document" in route_str or "rag" in route_str or "vector" in route_str:
        css_class, icon, label = "route-document", "📄", "DOCUMENT"
    elif "sql" in route_str or "database" in route_str or "db" in route_str:
        css_class, icon, label = "route-sql", "🗄️", "SQL"
    elif "hybrid" in route_str:
        css_class, icon, label = "route-hybrid", "🔗", "HYBRID"
    else:
        css_class, icon, label = "route-document", "📄", route_str.upper()
    
    st.markdown(f'<div class="route-badge {css_class}">{icon} Route: {label}</div>', unsafe_allow_html=True)


# ============================================================================
# Pages
# ============================================================================
def render_chat_page():
    """Main chat interface with complete chunk display."""
    st.markdown('<h2 style="margin-bottom: 0.5rem;">💬 Ask NorthStar Bank</h2>', unsafe_allow_html=True)
    st.caption("Ask about home loans, fixed deposits, credit cards, personal loans, and more.")
    
    # Chat history
    msg_idx = 0
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
            msg_idx += 1
        elif message["role"] == "assistant":
            with st.chat_message("assistant"):
                render_route_badge(message.get("route"))
                st.markdown(message["content"])
                render_citations(message.get("citations", []))
                render_retrieved_chunks(message.get("retrieved_chunks", []), msg_idx)
            msg_idx += 1
    
    # Chat input
    if prompt := st.chat_input("Ask about banking products, rates, eligibility...", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start = time.time()
                response = send_query(prompt, st.session_state.session_id)
                elapsed = time.time() - start
            
            render_route_badge(response.get("route"))
            st.markdown(response["answer"])
            st.caption(f"⏱️ {elapsed:.1f}s")
            render_citations(response.get("citations", []))
            render_retrieved_chunks(response.get("retrieved_chunks", []), msg_idx + 1)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"],
            "route": response.get("route"),
            "citations": response.get("citations", []),
            "retrieved_chunks": response.get("retrieved_chunks", []),
            "elapsed": elapsed,
        })


def render_upload_page():
    """Document upload interface."""
    st.markdown('<h2 style="margin-bottom: 0.5rem;">📤 Upload Knowledge Base</h2>', unsafe_allow_html=True)
    st.caption("Upload PDFs to ingest into the RAG pipeline. Documents are parsed, chunked, and embedded.")
    st.divider()
    
    uploaded_files = st.file_uploader(
        "Drag & drop PDF files here", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed"
    )
    
    if not uploaded_files:
        st.markdown("""
        <div class="upload-container">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📁</div>
            <div style="font-size: 1rem; color: #475569; font-weight: 500;">
                Drop PDF files here or click to browse
            </div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.5rem;">
                Supported: PDF files up to 50MB
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
        for f in uploaded_files:
            size_mb = f.size / (1024 * 1024)
            st.markdown(f"- `{f.name}` ({size_mb:.2f} MB)")
        st.divider()
        
        if st.button("🚀 Upload & Ingest", type="primary", use_container_width=True):
            progress = st.container()
            results = []
            
            for uploaded_file in uploaded_files:
                with progress:
                    status = st.empty()
                    bar = st.progress(0)
                    
                    status.markdown(f'<div class="status-card status-info">⏳ Uploading `{uploaded_file.name}`...</div>', unsafe_allow_html=True)
                    bar.progress(25)
                    
                    result = upload_document(uploaded_file, uploaded_file.name)
                    bar.progress(100)
                    results.append({"filename": uploaded_file.name, "result": result})
            
            st.divider()
            st.markdown("### Upload Results")
            
            for r in results:
                filename = r["filename"]
                result = r["result"]
                
                if result["success"]:
                    data = result["data"]
                    details = []
                    if isinstance(data, dict):
                        for label, key in [("Document ID", "document_id"), ("Parent Chunks", "parent_chunks"), 
                                          ("Child Chunks", "child_chunks"), ("Total", "total_stored"), ("Pages", "total_pages")]:
                            if key in data:
                                details.append(f"**{label}:** `{data[key]}`")
                    detail_str = " · ".join(details) if details else json.dumps(data, indent=2)
                    
                    st.markdown(f'<div class="status-card status-success">✅ <b>{filename}</b> — Ingested<br><span style="font-size:0.82rem;">{detail_str}</span></div>', unsafe_allow_html=True)
                else:
                    error = result.get("data", "Unknown error")
                    if isinstance(error, dict):
                        error = json.dumps(error, indent=2)
                    st.markdown(f'<div class="status-card status-error">❌ <b>{filename}</b> — Failed<br><span style="font-size:0.82rem;">{error}</span></div>', unsafe_allow_html=True)


# ============================================================================
# Main
# ============================================================================
def main():
    render_sidebar()
    
    if st.session_state.active_page == "chat":
        render_chat_page()
    elif st.session_state.active_page == "upload":
        render_upload_page()


if __name__ == "__main__":
    main()