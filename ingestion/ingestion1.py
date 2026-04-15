
# import os
# import uuid
# import asyncio
# import base64
# import io
# import json
# import re
# from datetime import datetime, timezone
# from typing import List, Dict, Any, Optional

# from docling.datamodel.base_models import InputFormat
# from docling.datamodel.pipeline_options import PdfPipelineOptions
# from docling.document_converter import DocumentConverter, PdfFormatOption
# from docling_core.types.doc import TableItem, PictureItem, TextItem
# from google.genai import types
# from google import genai
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_postgres import PGVector
# from sqlalchemy import create_engine, text
# from dotenv import load_dotenv

# load_dotenv(override=True)

# # ---------------------------------------------------------------------------
# # Constants
# # ---------------------------------------------------------------------------
# EMBEDDING_DIM = 1536
# EMBEDDING_MODEL = "gemini-embedding-2-preview"
# LLM_MODEL = os.getenv("GEMINI_MODEL")
# CHUNK_SIZE = 1500
# CHUNK_OVERLAP = 300

# # Edit these for your environment
# HARD_CODED_PDF_PATH = r"C:\Users\t91-labuser015568\Desktop\TCS_GEN_AI\multimodal-reranker-agentic-rag\data\RIL-Media-Release-RIL-Q2-FY2024-25-mini.pdf"

# TABLE_CAPTION_PROMPT = """\
# You are a financial document analyst. Given the following table data extracted from a PDF,
# generate a clear, concise caption that describes what the table contains, including key metrics,
# time periods, and units if present.

# Section: {section}
# Table Data:
# {table_data}

# Respond with ONLY the caption text, no JSON, no quotes. Keep it under 200 characters."""

# IMAGE_CAPTION_PROMPT = """\
# You are a financial document analyst. Given the section context and the image below,
# generate a clear, concise caption describing what the image/chart shows, including
# key data points, trends, or labels visible in the image.

# Section: {section}
# Docling Caption: {docling_caption}

# Respond with ONLY the caption text, no JSON, no quotes. Keep it under 200 characters."""


# # ---------------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------------
# def generate_document_metadata(doc_name: str, source_file: str,
#                                total_pages: int = 0,
#                                search_keywords: list[str] | None = None) -> Dict[str, Any]:
#     now = datetime.now(timezone.utc).isoformat()
#     return {
#         "document_id": str(uuid.uuid4()),
#         "document_name": doc_name,
#         "source_file": source_file,
#         "page": 0,
#         "total_pages": total_pages,
#         "ingested_at": now,
#         "updated_at": now,
#         "search_keywords": search_keywords or [],
#         "retrieval_weight": 1.0,
#     }


# def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
#     """Split text into overlapping character windows."""
#     chunks, start, step = [], 0, size - overlap
#     while start < len(text):
#         chunks.append(text[start:start + size])
#         start += step
#     return chunks


# # ---------------------------------------------------------------------------
# # Pipeline
# # ---------------------------------------------------------------------------
# class FinancialIngestionPipeline:
#     def __init__(self, db_connection_string: str, collection_name: str = "financial_rag"):
#         self.conn_string = db_connection_string
#         self.collection_name = collection_name

#         # Docling converter
#         opts = PdfPipelineOptions()
#         opts.do_ocr = True
#         opts.do_table_structure = True
#         opts.generate_picture_images = True
#         self.converter = DocumentConverter(
#             allowed_formats=[InputFormat.PDF],
#             format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)},
#         )

#         # ---- LangChain GoogleGenerativeAIEmbeddings (replaces direct API calls) ----
#         api_key = os.getenv("GEMINI_API_KEY")
#         if not api_key:
#             raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")

#         self.embeddings = GoogleGenerativeAIEmbeddings(
#             model=EMBEDDING_MODEL,
#             google_api_key=api_key,
#             output_dimensionality=1536
#         )

#         # PGVector store — now receives the embeddings object directly
#         self.vectorstore = PGVector(
#             embeddings=self.embeddings,
#             connection=self.conn_string,
#             collection_name=self.collection_name,
#             use_jsonb=True,
#         )

#         # Gemini client (kept only for LLM caption generation)
#         self.gemini = genai.Client(api_key=api_key)

#     # ---- LLM Caption Generation -------------------------------------------
#     async def generate_table_caption(self, table_data: str, section: str) -> str:
#         """Call Gemini LLM to generate a caption for table data."""
#         try:
#             prompt = TABLE_CAPTION_PROMPT.format(
#                 section=section or "Unknown",
#                 table_data=table_data[:3000],  # truncate to avoid token limits
#             )
#             resp = await asyncio.to_thread(
#                 self.gemini.models.generate_content,
#                 model=LLM_MODEL,
#                 contents=prompt,
#             )
#             return resp.text.strip()
#         except Exception as e:
#             print(f"[llm] table caption failed: {e}")
#             return ""

#     async def generate_image_caption(self, section: str, docling_caption: str,
#                                      img_base64: str) -> str:
#         """Call Gemini LLM (multimodal) to generate a caption for an image."""
#         try:
#             prompt = IMAGE_CAPTION_PROMPT.format(
#                 section=section or "Unknown",
#                 docling_caption=docling_caption or "No caption provided",
#             )

#             image_part = types.Part.from_bytes(data=base64.b64decode(img_base64), mime_type="image/png")

#             resp = await asyncio.to_thread(
#                 self.gemini.models.generate_content,
#                 model=LLM_MODEL,
#                 contents=[prompt, image_part],
#             )
#             return resp.text.strip()
#         except Exception as e:
#             print(f"[llm] image caption failed: {e}")
#             return ""

#     # ---- Table -> plain text -----------------------------------------------
#     @staticmethod
#     def table_to_text(node: TableItem, doc) -> str:
#         """Convert a TableItem to 'Header: value' rows."""
#         # Strategy 1 & 2: pandas DataFrame (try with and without doc arg)
#         if hasattr(node, "export_to_dataframe"):
#             for call_args in [(doc,), ()]:
#                 try:
#                     df = node.export_to_dataframe(*call_args)
#                     if df is not None and not df.empty:
#                         lines = []
#                         headers = [str(c).strip() for c in df.columns]
#                         for _, row in df.iterrows():
#                             pairs = [
#                                 f"{h}: {v}" for h, v in zip(headers, row)
#                                 if str(v).strip() not in ("", "nan", "None")
#                             ]
#                             if pairs:
#                                 lines.append("  |  ".join(pairs))
#                         if lines:
#                             return "\n".join(lines)
#                 except TypeError:
#                     continue
#                 except Exception as e:
#                     print(f"[table] export_to_dataframe failed: {e}")

#         # Strategy 3: HTML strip
#         if hasattr(node, "export_to_html"):
#             try:
#                 html = node.export_to_html(doc)
#                 if html:
#                     cleaned = re.sub(r"<[^>]+>", " ", html)
#                     cleaned = re.sub(r"\s+", " ", cleaned).strip()
#                     if cleaned:
#                         return cleaned
#             except Exception as e:
#                 print(f"[table] export_to_html failed: {e}")

#         # Strategy 4: raw grid data
#         grid = getattr(node, "data", None)
#         if grid and hasattr(grid, "grid") and grid.grid:
#             try:
#                 lines = []
#                 for row_cells in grid.grid:
#                     cells = [str(c).strip() for c in row_cells
#                              if str(c).strip() not in ("", "nan", "None")]
#                     if cells:
#                         lines.append("  |  ".join(cells))
#                 if lines:
#                     return "\n".join(lines)
#             except Exception as e:
#                 print(f"[table] grid access failed: {e}")

#         # Strategy 5: node.text fallback
#         fallback = getattr(node, "text", "").strip()
#         if fallback:
#             return fallback

#         print("[table] WARNING: all extraction strategies failed for table node")
#         return ""

#     # ---- Image -> base64 ---------------------------------------------------
#     @staticmethod
#     def image_to_base64(node: PictureItem, doc) -> Optional[str]:
#         """Extract picture as base64 PNG string."""
#         try:
#             pil_img = None
#             if hasattr(node, "get_image"):
#                 pil_img = node.get_image(doc)
#             elif hasattr(node, "image") and node.image:
#                 pil_img = getattr(node.image, "pil_image", None)
#             if pil_img:
#                 buf = io.BytesIO()
#                 pil_img.save(buf, format="PNG")
#                 return base64.b64encode(buf.getvalue()).decode()
#         except Exception:
#             pass
#         return None

#     # ---- Build JSON content for table/image --------------------------------
#     @staticmethod
#     def build_table_json(section: str, docling_caption: str,
#                          llm_caption: str, table_data: str) -> str:
#         """Build JSON content string for a table chunk."""
#         obj = {
#             "type": "table",
#             "section_header": section,
#             "docling_caption": docling_caption,
#             "llm_generated_caption": llm_caption,
#             "table_data": table_data,
#         }
#         return json.dumps(obj, ensure_ascii=False)

#     @staticmethod
#     def build_image_json(section: str, docling_caption: str,
#                          llm_caption: str, page_no: int) -> str:
#         """Build JSON content string for an image chunk."""
#         obj = {
#             "type": "image",
#             "section_header": section,
#             "docling_caption": docling_caption,
#             "llm_generated_caption": llm_caption,
#             "page": page_no,
#         }
#         return json.dumps(obj, ensure_ascii=False)

#     # ---- IVFFlat index (1536-dim) -----------------------------------------
#     def prepare_vector_column(self) -> None:
#         """Ensure embedding column is typed as vector(EMBEDDING_DIM).

#         MUST be called BEFORE inserting any data. If the column has the wrong
#         dimensionality (or no type), existing rows are deleted and the column
#         is re-typed so new inserts will succeed.
#         """
#         engine = create_engine(self.conn_string)
#         with engine.connect() as conn:
#             conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
#             conn.commit()

#             # Check current column dimensionality
#             row = conn.execute(text(
#                 "SELECT atttypmod FROM pg_attribute "
#                 "WHERE attrelid = 'langchain_pg_embedding'::regclass "
#                 "AND attname = 'embedding'"
#             )).fetchone()

#             current_dim = row[0] if row else -1
#             if current_dim != EMBEDDING_DIM:
#                 conn.execute(text("DELETE FROM langchain_pg_embedding"))
#                 conn.commit()
#                 print(f"[db] Cleared existing rows (had dim={current_dim}, need {EMBEDDING_DIM})")

#                 conn.execute(text(
#                     f"ALTER TABLE langchain_pg_embedding "
#                     f"ALTER COLUMN embedding TYPE vector({EMBEDDING_DIM}) "
#                     f"USING embedding::vector({EMBEDDING_DIM})"
#                 ))
#                 conn.commit()
#                 print(f"[db] Column set to vector({EMBEDDING_DIM})")
#             else:
#                 print(f"[db] Column already vector({EMBEDDING_DIM}) — OK")

#     def create_index(self, lists: int = 100) -> None:
#         """Create IVFFlat index on the embedding column.

#         Call AFTER inserting data. Only creates/recreates the index —
#         does NOT touch column type or data.
#         """
#         engine = create_engine(self.conn_string)
#         with engine.connect() as conn:
#             conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
#             conn.commit()

#             conn.execute(text("DROP INDEX IF EXISTS ivfflat_policy_idx"))
#             conn.commit()

#             conn.execute(text(
#                 f"CREATE INDEX IF NOT EXISTS ivfflat_policy_idx "
#                 f"ON langchain_pg_embedding "
#                 f"USING ivfflat (embedding vector_cosine_ops) "
#                 f"WITH (lists = {lists})"
#             ))
#             conn.commit()
#             print(f"[db] IVFFlat index ready (lists={lists}, dim={EMBEDDING_DIM})")

#     # ---- Parse PDF --------------------------------------------------------
#     def parse_document(self, file_path: str,
#                        base_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
#         """Parse a PDF into text / table / image chunks.

#         Tables and images are stored as JSON in the content field with:
#         - section_header: the owning section heading
#         - docling_caption: caption text extracted by Docling
#         - llm_generated_caption: AI-generated caption via Gemini LLM
#         - table_data / page: the actual content
#         """
#         result = self.converter.convert(file_path)
#         doc = result.document

#         base_meta["total_pages"] = len(doc.pages) if hasattr(doc, "pages") else 0
#         source_file = os.path.basename(file_path)
#         chunks: List[Dict[str, Any]] = []
#         section = None
#         last_caption = None
#         idx = 0

#         # Collect LLM tasks for tables and images (async batch)
#         llm_tasks: list = []

#         for item in doc.iterate_items():
#             node = item[0] if isinstance(item, tuple) else item

#             label = str(getattr(node, "label", "")).lower()
#             if label in ("page_header", "page_footer"):
#                 continue

#             prov = getattr(node, "prov", None)
#             page_no = prov[0].page_no + 1 if prov else 1
#             bbox = None
#             if prov and hasattr(prov[0], "bbox") and prov[0].bbox:
#                 b = prov[0].bbox
#                 bbox = [b.l, b.t, b.r, b.b]

#             def _meta(modality: str, elem_type: str, docling_label: str, img_b64=None):
#                 m = {**base_meta, "page": page_no, "chunk_index": idx,
#                      "modality": modality, "element_type": elem_type,
#                      "docling_label": docling_label,
#                      "section": section, "source_file": source_file,
#                      "page_number": page_no, "bbox": bbox,
#                      "image_base64": img_b64}
#                 return m

#             # --- Section headers & title ---
#             if "section_header" in label or label == "title":
#                 t = getattr(node, "text", "").strip()
#                 if t:
#                     section = t
#                     chunks.append({"content": t, "meta": _meta("text", label, label)})
#                     idx += 1

#             # --- Caption: remember for next table/image ---
#             elif label == "caption":
#                 t = getattr(node, "text", "").strip()
#                 if t:
#                     last_caption = t
#                     chunks.append({"content": t, "meta": _meta("text", "caption", "caption")})
#                     idx += 1

#             # --- Table: LLM caption + JSON content ---
#             elif isinstance(node, TableItem) or "table" in label:
#                 table_data = self.table_to_text(node, doc)
#                 if not table_data:
#                     print(f"[skip] table on page {page_no} produced no data")
#                     continue

#                 print(f"[table] page {page_no}: {len(table_data)} chars extracted")
#                 m = _meta("table", "table", "table")
#                 m["table_title"] = last_caption or section or ""
#                 m["section_header"] = section
#                 m["docling_caption"] = last_caption or ""

#                 # Placeholder chunk — will be filled after LLM batch
#                 chunks.append({
#                     "content": "",  # filled after LLM call
#                     "meta": m,
#                     "table_data": table_data,
#                     "section": section,
#                     "docling_caption": last_caption or "",
#                     "page_no": page_no,
#                 })
#                 llm_tasks.append(("table", len(chunks) - 1))
#                 idx += 1
#                 last_caption = None

#             # --- Image: LLM caption (multimodal) + JSON content ---
#             elif isinstance(node, PictureItem) or "picture" in label or "figure" in label or label == "chart":
#                 img_b64 = self.image_to_base64(node, doc)
#                 caption = (getattr(node, "text", "") or "").strip()
#                 print(f"[image] page {page_no}: extracted"
#                       + (f" ({len(img_b64)} bytes base64)" if img_b64 else " (no image data)"))

#                 m = _meta("image", "picture", "picture", img_b64)
#                 m["table_title"] = last_caption or section or ""
#                 m["section_header"] = section
#                 m["docling_caption"] = caption or last_caption or ""

#                 chunks.append({
#                     "content": "",  # filled after LLM call
#                     "meta": m,
#                     "img_base64": img_b64,
#                     "docling_caption": caption or last_caption or "",
#                     "section": section,
#                     "page_no": page_no,
#                 })
#                 llm_tasks.append(("image", len(chunks) - 1))
#                 idx += 1
#                 last_caption = None

#             # --- Text / paragraph / list_item / footnote ---
#             elif isinstance(node, TextItem) or label in (
#                 "text", "paragraph", "list_item", "footnote"
#             ):
#                 t = getattr(node, "text", "").strip()
#                 if not t:
#                     continue
#                 sub_chunks = split_text(t) if len(t) > CHUNK_SIZE else [t]
#                 for s in sub_chunks:
#                     chunks.append({"content": s, "meta": _meta("text", label, label)})
#                     idx += 1

#         # --- Run all LLM caption tasks in parallel ---
#         if llm_tasks:
#             print(f"[llm] generating captions for {len(llm_tasks)} tables/images...")
#             captions = self._run_llm_captions(chunks, llm_tasks)
#             for task_idx, (kind, chunk_idx) in enumerate(llm_tasks):
#                 chunk = chunks[chunk_idx]
#                 if kind == "table":
#                     chunk["content"] = self.build_table_json(
#                         section=chunk["section"],
#                         docling_caption=chunk["docling_caption"],
#                         llm_caption=captions[task_idx],
#                         table_data=chunk["table_data"],
#                     )
#                     print(f"[llm] table caption: {captions[task_idx][:80]}...")
#                 elif kind == "image":
#                     chunk["content"] = self.build_image_json(
#                         section=chunk["section"],
#                         docling_caption=chunk["docling_caption"],
#                         llm_caption=captions[task_idx],
#                         page_no=chunk["page_no"],
#                     )
#                     print(f"[llm] image caption: {captions[task_idx][:80]}...")

#         return chunks

#     def _run_llm_captions(self, chunks: list, tasks: list) -> list[str]:
#         """Run all LLM caption generation tasks asynchronously in parallel."""

#         async def _run():
#             async def _caption_for(task_idx: int, kind: str, chunk_idx: int):
#                 chunk = chunks[chunk_idx]
#                 if kind == "table":
#                     return await self.generate_table_caption(
#                         table_data=chunk["table_data"],
#                         section=chunk["section"],
#                     )
#                 elif kind == "image":
#                     if chunk.get("img_base64"):
#                         return await self.generate_image_caption(
#                             section=chunk["section"],
#                             docling_caption=chunk["docling_caption"],
#                             img_base64=chunk["img_base64"],
#                         )
#                     else:
#                         return chunk["docling_caption"] or ""
#                 return ""

#             results = await asyncio.gather(
#                 *[_caption_for(i, kind, idx) for i, (kind, idx) in enumerate(tasks)]
#             )
#             return results

#         return asyncio.run(_run())

#     # ---- Main entry -------------------------------------------------------
#     def process_and_store(self, file_path: str,
#                           base_meta: Dict[str, Any]) -> Dict[str, Any]:
#         """Prepare DB -> Parse -> LLM Captions -> Embed (via LangChain) -> Store -> Index."""
#         # Step 0: Ensure vector column has correct dimensionality BEFORE insert
#         self.prepare_vector_column()

#         # Step 1: Parse PDF + generate LLM captions
#         chunks = self.parse_document(file_path, base_meta)
#         print(f"Docling produced {len(chunks)} chunks")

#         # Step 2: Embed + Store in PGVector using GoogleGenerativeAIEmbeddings
#         #         add_texts() calls embed_documents() internally
#         texts = [c["content"] for c in chunks]
#         metas = [c["meta"] for c in chunks]
#         ids = [f"{m['document_id']}_{m['chunk_index']}" for m in metas]

#         self.vectorstore.add_texts(texts=texts, metadatas=metas, ids=ids)
#         print(f"Inserted {len(chunks)} chunks -> '{self.collection_name}' "
#               f"(embedded via GoogleGenerativeAIEmbeddings)")

#         # Step 3: Build IVFFlat index (data-safe, no row deletion)
#         self.create_index(lists=100)

#         return {
#             "status": "completed",
#             "document_id": base_meta["document_id"],
#             "indexed_chunks": len(chunks),
#             "total_pages": base_meta["total_pages"],
#         }


# # ---------------------------------------------------------------------------
# # Runner
# # ---------------------------------------------------------------------------
# def run_ingestion_pipeline(file_path: str, base_meta: Dict[str, Any]) -> Dict[str, Any]:
#     db_conn = os.getenv("DATABASE_URL")
#     if not db_conn:
#         raise EnvironmentError("DATABASE_URL environment variable is required.")
#     pipeline = FinancialIngestionPipeline(db_connection_string=db_conn)
#     return pipeline.process_and_store(file_path, base_meta)


# if __name__ == "__main__":
#     if not os.path.exists(HARD_CODED_PDF_PATH):
#         print(f"File not found: {HARD_CODED_PDF_PATH}")
#         exit(1)

#     print(f"Starting ingestion: {os.path.basename(HARD_CODED_PDF_PATH)}")
#     doc_name = os.path.basename(HARD_CODED_PDF_PATH)
#     base_meta = generate_document_metadata(
#         doc_name=doc_name,
#         source_file=HARD_CODED_PDF_PATH,
#         search_keywords=["Financial", "Report", doc_name.split(".")[0]],
#     )

#     try:
#         result = run_ingestion_pipeline(HARD_CODED_PDF_PATH, base_meta)
#         print(f"Ingestion completed!")
#         print(f"  Document ID : {result['document_id']}")
#         print(f"  Chunks      : {result['indexed_chunks']}")
#         print(f"  Pages       : {result['total_pages']}")
#     except Exception as e:
#         print(f"Ingestion failed: {e}")
#         import traceback
#         traceback.print_exc()
import os
import uuid
import asyncio
import base64
import io
import json
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from collections import OrderedDict
from PIL import Image as PILImage
from PIL import ImageOps  # For image resizing

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from google.genai import types
from google import genai
from langchain_postgres import PGVector
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# LangChain Splitters
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 1536
EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = os.getenv("GEMINI_MODEL")

# Image compression settings (keep base64 under ~500KB per image for JSONB safety)
MAX_IMAGE_DIMENSION = 1024      # Max width/height in pixels (resize larger images)
JPEG_QUALITY = 75               # JPEG compression quality (1-100)
MAX_IMAGE_BASE64_BYTES = 500_000  # ~500KB limit before aggressive resize

# Parent-Child Chunking Strategy
#   Parent: full section content (large, NOT embedded, used as retrieval context)
#   Child:  small size-split pieces (embedded, used for similarity search)
PARENT_MAX_SIZE = 3000    # Max chars for a parent chunk before forced split
CHILD_CHUNK_SIZE = 500    # Smaller children = more precise retrieval matches
CHILD_CHUNK_OVERLAP = 100

HARD_CODED_PDF_PATH = r"C:\Users\t91-labuser015568\Desktop\TCS_GEN_AI\multimodal-reranker-agentic-rag\data\KB_Smart_Banking.pdf"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def generate_document_metadata(doc_name: str, source_file: str,
                               total_pages: int = 0,
                               search_keywords: list[str] | None = None) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "document_id": str(uuid.uuid4()),
        "document_name": doc_name,
        "source_file": source_file,
        "page": 0,
        "total_pages": total_pages,
        "ingested_at": now,
        "updated_at": now,
        "search_keywords": search_keywords or [],
        "retrieval_weight": 1.0,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class FinancialIngestionPipeline:
    def __init__(self, db_connection_string: str, collection_name: str = "financial_rag"):
        self.conn_string = db_connection_string
        self.collection_name = collection_name

        # Docling converter
        opts = PdfPipelineOptions()
        opts.do_ocr = True
        opts.do_table_structure = True
        opts.generate_picture_images = True
        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)},
        )

        # PGVector store (we'll pass embeddings manually via add_embeddings)
        self.vectorstore = PGVector(
            embeddings=None,
            connection=self.conn_string,
            collection_name=self.collection_name,
            use_jsonb=True,
        )

        # Gemini client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")
        self.gemini = genai.Client(api_key=api_key)

    # ---- Embedding --------------------------------------------------------
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using Gemini (1536-dim)."""
        async def _embed_one(content: str) -> list[float]:
            try:
                resp = self.gemini.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=[content],
                    config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
                )
                return resp.embeddings[0].values
            except Exception as e:
                print(f"[embed] failed: {e}")
                return [0.0] * EMBEDDING_DIM

        return await asyncio.gather(*[_embed_one(t) for t in texts])

    # ---- Parse PDF (Element-level + Parent-Child Chunking) -----------------
    def parse_document(self, file_path: str, base_meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse PDF -> Docling elements -> Section group -> Parent-Child split
        -> Metadata enrichment with page numbers, modalities, table data,
        base64 images, and parent-child relationships.

        PARENT-CHILD STRATEGY:
        ┌─────────────────────────────────────────────────────────┐
        │ PARENT CHUNK (is_parent=true)                           │
        │  - Full section content (up to PARENT_MAX_SIZE chars)   │
        │  - NOT embedded (stored with zero vector)               │
        │  - Contains: table_data, image_base64, full text        │
        │  - Serves as the retrieval context for the LLM          │
        ├─────────────────────────────────────────────────────────┤
        │ CHILD CHUNKS (is_parent=false)                          │
        │  - Small pieces of parent (CHILD_CHUNK_SIZE chars)      │
        │  - EMBEDDED and used for similarity search              │
        │  - Each child has parent_chunk_id -> links to parent    │
        │  - Parent has child_chunk_ids -> lists all children     │
        │  - Search children -> match -> pull parent -> LLM       │
        └─────────────────────────────────────────────────────────┘

        WHY: Child chunks give precise vector matches (small = focused embedding).
             But the LLM needs the full section (rate table, eligibility list,
             charge schedule) to answer accurately. Parent provides that context.
        """
        result = self.converter.convert(file_path)
        doc = result.document
        base_meta["total_pages"] = len(doc.pages) if hasattr(doc, "pages") else 0

        # =====================================================================
        # STEP 1: Extract elements from Docling document
        # Strategy A: Try structured element iteration (version-compatible)
        # Strategy B: Fall back to doc.export_to_markdown() if A yields nothing
        # =====================================================================
        raw_elements: List[Dict[str, Any]] = []
        current_headers = {"H1": "", "H2": "", "H3": ""}

        # ── Helper: extract text from an item across all Docling versions ──
        def _extract_text(item) -> str:
            """Try every known text-access pattern on a Docling element."""
            # Pattern 1: item.content.text  (DoclingDocumentContentItem)
            if hasattr(item, "content") and item.content is not None:
                if isinstance(item.content, str):
                    return item.content.strip()
                if hasattr(item.content, "text"):
                    return (item.content.text or "").strip()
            # Pattern 2: item.text  (some ContentItem subclasses expose it directly)
            if hasattr(item, "text") and item.text is not None:
                if isinstance(item.text, str):
                    return item.text.strip()
            return ""

        # ── Helper: get page number from Docling provenance ──
        def _get_page_no(item) -> int:
            if hasattr(item, "prov") and item.prov and len(item.prov) > 0:
                return item.prov[0].page_no + 1
            if hasattr(item, "self_ref") and item.self_ref:
                sr = str(item.self_ref)
                # self_ref format like "#/pages/3/elements/42" -> page 4
                m = re.search(r'/pages/(\d+)', sr)
                if m:
                    return int(m.group(1)) + 1
            return 0

        # ── Helper: extract caption from item ──
        def _get_caption(item) -> str:
            if hasattr(item, "caption") and item.caption:
                if hasattr(item.caption, "text"):
                    return (item.caption.text or "").strip()
                return str(item.caption).strip()
            return ""

        # ── Helper: convert PIL Image to base64 with compression ──
        def _pil_to_base64(pil_img: PILImage.Image) -> Optional[str]:
            """Convert PIL Image to JPEG base64. Resizes if too large."""
            try:
                img = pil_img.copy()
                # Convert RGBA/LA/P to RGB (JPEG doesn't support alpha)
                if img.mode in ("RGBA", "LA", "P"):
                    bg = PILImage.new("RGB", img.size, (255, 255, 255))
                    if img.mode == "P":
                        img = img.convert("RGBA")
                    bg.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                    img = bg
                elif img.mode != "RGB":
                    img = img.convert("RGB")

                # Resize if dimension exceeds max
                if img.width > MAX_IMAGE_DIMENSION or img.height > MAX_IMAGE_DIMENSION:
                    img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), PILImage.LANCZOS)

                # Try JPEG with decreasing quality until under limit
                for quality in [JPEG_QUALITY, 50, 30, 15]:
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=quality, optimize=True)
                    img_bytes = buf.getvalue()
                    if len(img_bytes) <= MAX_IMAGE_BASE64_BYTES:
                        return base64.b64encode(img_bytes).decode("utf-8")

                # Last resort: very aggressive resize + low quality
                img.thumbnail((512, 512), PILImage.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=10, optimize=True)
                img_bytes = buf.getvalue()
                return base64.b64encode(img_bytes).decode("utf-8")
            except Exception as e:
                print(f"[image] base64 conversion failed: {e}")
                return None

        # ── Helper: check if an item is an image (across all Docling versions) ──
        def _is_image_item(item) -> bool:
            """Broad image detection across all Docling versions."""
            label = getattr(item, "label", "") or ""
            # Direct label matches
            if label in ("picture", "image", "PictureItem", "FigureItem", "Figure"):
                return True
            # Class name check (e.g. docling.document_converter.PictureItem)
            cls_name = getattr(item.__class__, "__name__", "") or ""
            if "Picture" in cls_name or "Figure" in cls_name or "Image" in cls_name:
                return True
            # Has an image attribute with actual PIL Image data
            if hasattr(item, "image") and item.image is not None:
                if isinstance(item.image, PILImage.Image):
                    return True
            # docling_type attribute
            dt = getattr(item, "docling_type", "") or ""
            if "picture" in dt.lower() or "figure" in dt.lower() or "image" in dt.lower():
                return True
            return False

        # ── Helper: extract PIL image from any item/docling picture ──
        def _extract_pil_image(item_or_picture) -> Optional[PILImage.Image]:
            """Try every known pattern to get a PIL Image."""
            obj = item_or_picture
            # Direct .image attribute
            if hasattr(obj, 'image') and obj.image is not None:
                if isinstance(obj.image, PILImage.Image):
                    return obj.image
            # Object itself is a PIL Image
            if isinstance(obj, PILImage.Image):
                return obj
            return None

        # ─────────────────────────────────────────────────────────────
        # STRATEGY A: Element-level iteration
        # ─────────────────────────────────────────────────────────────
        try:
            _iterator = None
            if hasattr(doc, 'iterate_items'):
                _iterator = doc.iterate_items()
            elif hasattr(doc, 'body') and hasattr(doc.body, 'iterate_items'):
                _iterator = doc.body.iterate_items()
            elif hasattr(doc, 'body') and hasattr(doc.body, 'children'):
                def _walk(node, level=0):
                    yield node, level
                    if hasattr(node, 'children') and node.children:
                        for child in node.children:
                            yield from _walk(child, level + 1)
                _iterator = _walk(doc.body)

            if _iterator is not None:
                for item, level in _iterator:
                    page_no = _get_page_no(item)
                    text = _extract_text(item)
                    item_label = getattr(item, "label", "") or ""

                    # Skip group/root nodes that have no text of their own
                    if not text and item_label in ("", "group", "root", "body"):
                        continue
                    if not text:
                        continue

                    modality = "text"
                    element_type = "text"
                    extra_meta: Dict[str, Any] = {}

                    # --- Section headers ---
                    if item_label == "section_header":
                        try:
                            hl = int(getattr(item, "heading_level", level))
                            if hl <= 0:
                                current_headers["H1"] = text
                            elif hl == 1:
                                current_headers["H2"] = text
                            elif hl >= 2:
                                current_headers["H3"] = text
                        except Exception:
                            if level <= 0:
                                current_headers["H1"] = text
                            elif level == 1:
                                current_headers["H2"] = text
                            else:
                                current_headers["H3"] = text

                    # --- Tables ---
                    elif item_label == "table" or (hasattr(item, "data") and item.data is not None):
                        modality = "table"
                        element_type = "table"
                        if hasattr(item, "data") and item.data is not None:
                            for _method in ("to_markdown", "export_to_markdown"):
                                try:
                                    extra_meta["table_data"] = getattr(item.data, _method)()
                                    break
                                except Exception:
                                    pass
                            if "table_data" not in extra_meta:
                                extra_meta["table_data"] = str(item.data)
                        caption = _get_caption(item)
                        if caption:
                            extra_meta["docling_caption"] = caption

                    # --- Images (broad detection) ---
                    elif _is_image_item(item):
                        modality = "image"
                        element_type = "image"
                        try:
                            # Try doc.get_picture first, then item directly
                            pic_data = None
                            if hasattr(doc, 'get_picture'):
                                try:
                                    pic_data = doc.get_picture(item)
                                except Exception:
                                    pass
                            if pic_data is None:
                                pic_data = item

                            pil_image = _extract_pil_image(pic_data)

                            if pil_image is not None:
                                b64_str = _pil_to_base64(pil_image)
                                if b64_str:
                                    extra_meta["image_base64"] = b64_str
                                    extra_meta["image_mime_type"] = "image/jpeg"
                                    extra_meta["image_size_bytes"] = len(base64.b64decode(b64_str))
                                    extra_meta["image_width"] = pil_image.width
                                    extra_meta["image_height"] = pil_image.height
                                    print(f"  [image] Extracted image: {pil_image.width}x{pil_image.height} -> {extra_meta['image_size_bytes']} bytes")
                                else:
                                    print(f"  [image] Failed to encode image to base64")
                            else:
                                print(f"  [image] Could not extract PIL Image from item (label={item_label})")
                        except Exception as e:
                            print(f"  [parse] image extract failed: {e}")

                        caption = _get_caption(item)
                        if caption:
                            extra_meta["docling_caption"] = caption
                            text = caption if caption else "[Image]"

                    section = " > ".join(
                        h for h in [
                            current_headers["H1"],
                            current_headers["H2"],
                            current_headers["H3"],
                        ] if h
                    ) or "Unknown"

                    raw_elements.append({
                        "text": text,
                        "page_no": page_no,
                        "modality": modality,
                        "element_type": element_type,
                        "section": section,
                        "headers": dict(current_headers),
                        "extra_meta": extra_meta,
                    })

        except Exception as e:
            print(f"[parse] Strategy A (element iteration) failed: {e}")

        # ─────────────────────────────────────────────────────────────
        # STRATEGY B: Markdown export fallback
        # When element iteration yields nothing, fall back to the
        # reliable doc.export_to_markdown() API.  We lose per-element
        # page numbers but gain guaranteed content extraction.
        # We supplement with doc.tables / doc.pictures for modality.
        # ─────────────────────────────────────────────────────────────
        if not raw_elements:
            print("[parse] Strategy A yielded 0 elements. Trying Strategy B (markdown export)...")
            current_headers = {"H1": "", "H2": "", "H3": ""}

            try:
                md_text = doc.export_to_markdown()
            except Exception:
                md_text = ""

            if md_text and md_text.strip():
                # Parse markdown into elements by splitting on headings
                running_text_lines: List[str] = []
                current_element_type = "text"

                def _flush_running():
                    """Push accumulated text lines as one element."""
                    nonlocal running_text_lines, current_element_type
                    block = "\n".join(running_text_lines).strip()
                    running_text_lines = []
                    if not block:
                        return
                    section = " > ".join(
                        h for h in [
                            current_headers["H1"],
                            current_headers["H2"],
                            current_headers["H3"],
                        ] if h
                    ) or "Unknown"
                    raw_elements.append({
                        "text": block,
                        "page_no": 0,
                        "modality": current_element_type,
                        "element_type": current_element_type,
                        "section": section,
                        "headers": dict(current_headers),
                        "extra_meta": {},
                    })

                for line in md_text.split("\n"):
                    stripped = line.strip()
                    if not stripped:
                        continue

                    # Detect markdown headings
                    heading_match = re.match(r'^(#{1,6})\s+(.*)', stripped)
                    if heading_match:
                        _flush_running()  # flush previous block
                        h_level = len(heading_match.group(1))
                        h_text = heading_match.group(2).strip()
                        current_element_type = "section_header"

                        if h_level == 1:
                            current_headers["H1"] = h_text
                            current_headers["H2"] = ""
                            current_headers["H3"] = ""
                        elif h_level == 2:
                            current_headers["H2"] = h_text
                            current_headers["H3"] = ""
                        elif h_level >= 3:
                            current_headers["H3"] = h_text

                        section = " > ".join(
                            h for h in [
                                current_headers["H1"],
                                current_headers["H2"],
                                current_headers["H3"],
                            ] if h
                        ) or "Unknown"
                        raw_elements.append({
                            "text": h_text,
                            "page_no": 0,
                            "modality": "text",
                            "element_type": "section_header",
                            "section": section,
                            "headers": dict(current_headers),
                            "extra_meta": {},
                        })
                        current_element_type = "text"
                    else:
                        # Detect table rows (lines starting with | )
                        if stripped.startswith("|"):
                            if current_element_type != "table":
                                _flush_running()
                                current_element_type = "table"
                        else:
                            if current_element_type == "table":
                                _flush_running()
                                current_element_type = "text"
                        running_text_lines.append(stripped)

                _flush_running()  # flush last block

            # ── Supplement: extract tables from doc.tables ──
            try:
                doc_tables = getattr(doc, "tables", None)
                if doc_tables is None:
                    doc_tables = getattr(doc, "_tables", None)
                if doc_tables and hasattr(doc_tables, 'values'):
                    for tkey, titem in doc_tables.items():
                        tdata = None
                        if hasattr(titem, "data") and titem.data is not None:
                            for _m in ("to_markdown", "export_to_markdown"):
                                try:
                                    tdata = getattr(titem.data, _m)()
                                    break
                                except Exception:
                                    pass
                            if tdata is None:
                                tdata = str(titem.data)
                        caption = _get_caption(titem) if hasattr(titem, '__dict__') else ""
                        page_no = _get_page_no(titem) if hasattr(titem, '__dict__') else 0
                        raw_elements.append({
                            "text": caption or tdata or "",
                            "page_no": page_no,
                            "modality": "table",
                            "element_type": "table",
                            "section": "Unknown",
                            "headers": dict(current_headers),
                            "extra_meta": {
                                "table_data": tdata,
                                "docling_caption": caption,
                            } if caption else {"table_data": tdata} if tdata else {},
                        })
            except Exception as e:
                print(f"[parse] Strategy B table extraction failed: {e}")

            # ── Supplement: extract pictures from doc.pictures ──
            try:
                doc_pics = getattr(doc, "pictures", None)
                if doc_pics is None:
                    doc_pics = getattr(doc, "_pictures", None)
                if doc_pics and isinstance(doc_pics, dict):
                    for pkey, pitem in doc_pics.items():
                        pil_image = None
                        if hasattr(pitem, 'image') and pitem.image is not None:
                            pil_image = pitem.image
                        elif isinstance(pitem, PILImage.Image):
                            pil_image = pitem

                        caption = _get_caption(pitem) if hasattr(pitem, '__dict__') else ""
                        page_no = _get_page_no(pitem) if hasattr(pitem, '__dict__') else 0

                        if pil_image is not None:
                            try:
                                buf = io.BytesIO()
                                pil_image.save(buf, format="PNG")
                                img_bytes = buf.getvalue()
                                raw_elements.append({
                                    "text": caption or "[Image]",
                                    "page_no": page_no,
                                    "modality": "image",
                                    "element_type": "image",
                                    "section": "Unknown",
                                    "headers": dict(current_headers),
                                    "extra_meta": {
                                        "image_base64": base64.b64encode(img_bytes).decode("utf-8"),
                                        "image_mime_type": "image/png",
                                        "image_size_bytes": len(img_bytes),
                                        "docling_caption": caption,
                                    },
                                })
                            except Exception as e:
                                print(f"[parse] Strategy B image extract failed: {e}")
                        elif caption:
                            raw_elements.append({
                                "text": caption,
                                "page_no": page_no,
                                "modality": "image",
                                "element_type": "image",
                                "section": "Unknown",
                                "headers": dict(current_headers),
                                "extra_meta": {"docling_caption": caption},
                            })
            except Exception as e:
                print(f"[parse] Strategy B picture extraction failed: {e}")

        if not raw_elements:
            raise ValueError("No elements extracted. Check PDF quality or Docling config.")

        # =====================================================================
        # STEP 1.5: GUARANTEED image sweep (runs ALWAYS, not just in Strategy B)
        # =====================================================================
        # Even if Strategy A found text/table elements, it may have missed images.
        # doc.pictures is the most reliable way to get ALL images regardless of
        # how element iteration labels them.  We de-duplicate by checking if an
        # image with the same page + approximate caption already exists.
        # ─────────────────────────────────────────────────────────────────
        _existing_image_pages = {(e["page_no"], e["text"][:50]) for e in raw_elements if e["modality"] == "image"}
        _images_found_by_sweep = 0

        for _pics_attr in ("pictures", "_pictures"):
            doc_pics = getattr(doc, _pics_attr, None)
            if doc_pics is None:
                continue
            # pictures can be a dict or a list
            if isinstance(doc_pics, dict):
                _pic_iterable = doc_pics.values()
            elif isinstance(doc_pics, (list, tuple)):
                _pic_iterable = doc_pics
            else:
                try:
                    _pic_iterable = list(doc_pics)
                except Exception:
                    _pic_iterable = []

            for pitem in _pic_iterable:
                try:
                    pil_image = _extract_pil_image(pitem)
                    caption = _get_caption(pitem) if hasattr(pitem, '__dict__') else ""
                    page_no = _get_page_no(pitem) if hasattr(pitem, '__dict__') else 0

                    # Skip self_ref items that aren't actual picture objects
                    if pil_image is None and not caption:
                        continue

                    # De-dup check
                    _dedup_key = (page_no, (caption or "[Image]")[:50])
                    if _dedup_key in _existing_image_pages:
                        continue

                    if pil_image is not None:
                        b64_str = _pil_to_base64(pil_image)
                        if b64_str:
                            raw_elements.append({
                                "text": caption or "[Image]",
                                "page_no": page_no,
                                "modality": "image",
                                "element_type": "image",
                                "section": "Unknown",
                                "headers": dict(current_headers),
                                "extra_meta": {
                                    "image_base64": b64_str,
                                    "image_mime_type": "image/jpeg",
                                    "image_size_bytes": len(base64.b64decode(b64_str)),
                                    "image_width": pil_image.width,
                                    "image_height": pil_image.height,
                                    "docling_caption": caption,
                                },
                            })
                            _existing_image_pages.add(_dedup_key)
                            _images_found_by_sweep += 1
                            print(f"  [image-sweep] Page {page_no}: {pil_image.width}x{pil_image.height} -> {len(base64.b64decode(b64_str))} bytes")
                    elif caption:
                        raw_elements.append({
                            "text": caption,
                            "page_no": page_no,
                            "modality": "image",
                            "element_type": "image",
                            "section": "Unknown",
                            "headers": dict(current_headers),
                            "extra_meta": {"docling_caption": caption},
                        })
                        _existing_image_pages.add(_dedup_key)
                        _images_found_by_sweep += 1
                except Exception as e:
                    print(f"  [image-sweep] Failed for one picture: {e}")
            break  # Stop after first successful attribute

        print(
            f"[parse] Extracted {len(raw_elements)} elements: "
            f"text={sum(1 for e in raw_elements if e['modality']=='text')}, "
            f"table={sum(1 for e in raw_elements if e['modality']=='table')}, "
            f"image={sum(1 for e in raw_elements if e['modality']=='image')}"
            f"{_images_found_by_sweep > 0 and f' (+{_images_found_by_sweep} from picture sweep)' or ''}"
        )

        # =====================================================================
        # STEP 2: Group elements by section
        # =====================================================================
        section_groups: OrderedDict[str, list] = OrderedDict()
        for elem in raw_elements:
            key = elem["section"]
            if key not in section_groups:
                section_groups[key] = []
            section_groups[key].append(elem)

        # =====================================================================
        # STEP 3: Build Parent-Child chunks
        # =====================================================================
        parent_chunks: List[Dict[str, Any]] = []   # Large context chunks (not embedded)
        child_chunks: List[Dict[str, Any]] = []    # Small retrieval chunks (embedded)

        parent_idx = 0
        child_idx = 0

        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHILD_CHUNK_SIZE,
            chunk_overlap=CHILD_CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        for section_name, elems in section_groups.items():
            # ── Separate images from text/table elements ──
            non_image_elems = [e for e in elems if e["modality"] != "image"]
            image_elems = [e for e in elems if e["modality"] == "image"]

            # ── Build full section text for the PARENT ──
            section_parts: List[str] = []
            section_tables: List[str] = []      # Collect all table data for parent
            section_captions: List[str] = []    # Collect captions
            section_images: List[Dict] = []     # Collect image metadata

            for elem in non_image_elems:
                section_parts.append(elem["text"])
                if "table_data" in elem.get("extra_meta", {}):
                    section_tables.append(elem["extra_meta"]["table_data"])
                if "docling_caption" in elem.get("extra_meta", {}):
                    section_captions.append(elem["extra_meta"]["docling_caption"])

            for img_elem in image_elems:
                if img_elem["extra_meta"].get("docling_caption"):
                    section_captions.append(img_elem["extra_meta"]["docling_caption"])
                    section_parts.append(img_elem["extra_meta"]["docling_caption"])
                if img_elem["extra_meta"].get("image_base64"):
                    section_images.append({
                        "image_base64": img_elem["extra_meta"]["image_base64"],
                        "image_mime_type": img_elem["extra_meta"].get("image_mime_type", "image/png"),
                        "image_size_bytes": img_elem["extra_meta"].get("image_size_bytes", 0),
                        "docling_caption": img_elem["extra_meta"].get("docling_caption", ""),
                        "page": img_elem["page_no"],
                    })

            full_section_text = "\n\n".join(section_parts)
            if not full_section_text.strip():
                continue

            # ── Determine modality for the parent (takes the "richest" type) ──
            parent_modality = "text"
            if section_tables:
                parent_modality = "table"
            if section_images:
                parent_modality = "image" if not section_tables else "mixed"

            # ── Compute page range for parent ──
            all_pages = [e["page_no"] for e in elems if e["page_no"] > 0]
            min_page = min(all_pages) if all_pages else 0
            max_page = max(all_pages) if all_pages else 0

            headers = elems[0]["headers"]

            # ── If section is too large, split into multiple parents ──
            if len(full_section_text) > PARENT_MAX_SIZE:
                parent_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=PARENT_MAX_SIZE,
                    chunk_overlap=200,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )
                parent_docs = parent_splitter.split_documents(
                    [Document(page_content=full_section_text, metadata={})]
                )
            else:
                parent_docs = [Document(page_content=full_section_text, metadata={})]

            for p_doc_idx, p_doc in enumerate(parent_docs):
                parent_id = f"{base_meta['document_id']}_p{parent_idx}"

                # ── Build PARENT metadata ──
                parent_meta: Dict[str, Any] = {
                    **base_meta,
                    "chunk_index": parent_idx,
                    "chunk_level": "parent",
                    "is_parent": True,
                    "parent_chunk_id": None,       # Parents have no parent
                    "child_chunk_ids": [],          # Populated after children are created
                    "modality": parent_modality,
                    "element_type": parent_modality,
                    "section": section_name,
                    "page": min_page if p_doc_idx == 0 else min_page + p_doc_idx,
                    "page_start": min_page,
                    "page_end": max_page,
                    "header_metadata": headers,
                    "source_file": os.path.basename(file_path),
                }

                # Attach all table data to first parent (preserves full tables)
                if p_doc_idx == 0 and section_tables:
                    parent_meta["table_data"] = "\n\n".join(section_tables)
                if p_doc_idx == 0 and section_captions:
                    parent_meta["docling_caption"] = " | ".join(section_captions)
                if p_doc_idx == 0 and section_images:
                    parent_meta["section_images"] = section_images
                    parent_meta["image_count"] = len(section_images)

                parent_chunks.append({
                    "content": p_doc.page_content,
                    "meta": parent_meta,
                    "parent_id": parent_id,
                })

                # ── Build CHILD chunks from this parent ──
                child_docs = child_splitter.split_documents(
                    [Document(page_content=p_doc.page_content, metadata={})]
                )

                child_ids_for_this_parent: List[str] = []

                for c_doc in child_docs:
                    child_id = f"{base_meta['document_id']}_c{child_idx}"
                    child_ids_for_this_parent.append(child_id)

                    # Match child to its best source element for modality
                    child_words = set(c_doc.page_content.lower().split())
                    best_elem = max(
                        non_image_elems,
                        key=lambda e: len(child_words & set(e["text"].lower().split())),
                        default=non_image_elems[0] if non_image_elems else elems[0],
                    )

                    child_meta: Dict[str, Any] = {
                        **base_meta,
                        "chunk_index": child_idx,
                        "chunk_level": "child",
                        "is_parent": False,
                        "parent_chunk_id": parent_id,    # Link to parent
                        "child_chunk_ids": [],             # Children have no children
                        "modality": best_elem["modality"],
                        "element_type": best_elem["element_type"],
                        "section": section_name,
                        "page": best_elem["page_no"],
                        "page_start": min_page,
                        "page_end": max_page,
                        "header_metadata": headers,
                        "source_file": os.path.basename(file_path),
                    }

                    # Propagate element-specific metadata (table_data, caption, etc.)
                    for k, v in best_elem["extra_meta"].items():
                        child_meta[k] = v

                    # If this section has images, attach image references to child too
                    # so the RAG agent knows the parent has images when pulling context
                    if section_images and p_doc_idx == 0:
                        child_meta["has_section_images"] = True
                        child_meta["section_image_count"] = len(section_images)
                        # Only attach captions to children (NOT base64 — too large for child metadata)
                        child_meta["section_image_captions"] = [
                            img.get("docling_caption", "") for img in section_images
                        ]

                    child_chunks.append({
                        "content": c_doc.page_content,
                        "meta": child_meta,
                        "child_id": child_id,
                    })
                    child_idx += 1

                # ── Wire up parent -> children ──
                parent_meta["child_chunk_ids"] = child_ids_for_this_parent
                parent_idx += 1

            # ── Image-only elements: each becomes both parent AND child ──
            for img_elem in image_elems:
                img_parent_id = f"{base_meta['document_id']}_p{parent_idx}"
                img_child_id = f"{base_meta['document_id']}_c{child_idx}"

                caption = img_elem["extra_meta"].get("docling_caption", "[Image - no caption]")

                # Image parent (carries the base64 image)
                img_parent_meta: Dict[str, Any] = {
                    **base_meta,
                    "chunk_index": parent_idx,
                    "chunk_level": "parent",
                    "is_parent": True,
                    "parent_chunk_id": None,
                    "child_chunk_ids": [img_child_id],
                    "modality": "image",
                    "element_type": "image",
                    "section": img_elem["section"],
                    "page": img_elem["page_no"],
                    "page_start": img_elem["page_no"],
                    "page_end": img_elem["page_no"],
                    "header_metadata": img_elem["headers"],
                    "source_file": os.path.basename(file_path),
                }
                for k, v in img_elem["extra_meta"].items():
                    img_parent_meta[k] = v

                parent_chunks.append({
                    "content": caption,
                    "meta": img_parent_meta,
                    "parent_id": img_parent_id,
                })

                # Image child (caption text for embedding/search)
                img_child_meta: Dict[str, Any] = {
                    **base_meta,
                    "chunk_index": child_idx,
                    "chunk_level": "child",
                    "is_parent": False,
                    "parent_chunk_id": img_parent_id,
                    "child_chunk_ids": [],
                    "modality": "image",
                    "element_type": "image",
                    "section": img_elem["section"],
                    "page": img_elem["page_no"],
                    "page_start": img_elem["page_no"],
                    "page_end": img_elem["page_no"],
                    "header_metadata": img_elem["headers"],
                    "source_file": os.path.basename(file_path),
                }
                if "docling_caption" in img_elem["extra_meta"]:
                    img_child_meta["docling_caption"] = img_elem["extra_meta"]["docling_caption"]

                child_chunks.append({
                    "content": caption,
                    "meta": img_child_meta,
                    "child_id": img_child_id,
                })
                parent_idx += 1
                child_idx += 1

        # ── Summary ──
        _parents_with_images = sum(1 for p in parent_chunks if p["meta"].get("section_images") or p["meta"].get("image_base64"))
        _parents_with_tables = sum(1 for p in parent_chunks if p["meta"].get("table_data"))
        _image_parents = sum(1 for p in parent_chunks if p["meta"].get("modality") == "image")

        print(
            f"\n[parse] Parent-Child chunking complete:"
            f"\n  Parents:  {len(parent_chunks)} (large context chunks, NOT embedded)"
            f"\n  Children: {len(child_chunks)} (small retrieval chunks, embedded)"
            f"\n  Pages:    {base_meta['total_pages']}"
            f"\n  Sections: {len(section_groups)}"
            f"\n  Parents with images: {_parents_with_images} (section_images) + {_image_parents} (image-only)"
            f"\n  Parents with tables: {_parents_with_tables}"
        )

        avg_children = len(child_chunks) / max(len(parent_chunks), 1)
        print(f"  Avg children per parent: {avg_children:.1f}")

        return {
            "parent_chunks": parent_chunks,
            "child_chunks": child_chunks,
        }

    # ---- IVFFlat index (1536-dim) -----------------------------------------
    def prepare_vector_column(self) -> None:
        engine = create_engine(self.conn_string)
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            row = conn.execute(text(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = 'langchain_pg_embedding'::regclass AND attname = 'embedding'"
            )).fetchone()
            current_dim = row[0] if row else -1
            if current_dim != EMBEDDING_DIM:
                conn.execute(text("DELETE FROM langchain_pg_embedding"))
                conn.commit()
                conn.execute(text(
                    f"ALTER TABLE langchain_pg_embedding ALTER COLUMN embedding TYPE vector({EMBEDDING_DIM}) USING embedding::vector({EMBEDDING_DIM})"
                ))
                conn.commit()
                print(f"[db] Column set to vector({EMBEDDING_DIM})")
            else:
                print(f"[db] Column already vector({EMBEDDING_DIM}) -- OK")

    def create_index(self, lists: int = 100) -> None:
        engine = create_engine(self.conn_string)
        with engine.connect() as conn:
            conn.execute(text("DROP INDEX IF EXISTS ivfflat_policy_idx"))
            conn.commit()
            conn.execute(text(
                f"CREATE INDEX IF NOT EXISTS ivfflat_policy_idx "
                f"ON langchain_pg_embedding USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists})"
            ))
            conn.commit()
            print(f"[db] IVFFlat index ready (lists={lists})")

    # ---- Main entry (Parent-Child aware) ----------------------------------
    def process_and_store(self, file_path: str, base_meta: Dict[str, Any]) -> Dict[str, Any]:
        self.prepare_vector_column()

        # 1. Parse -> parent + child chunks
        result = self.parse_document(file_path, base_meta)
        parent_chunks = result["parent_chunks"]
        child_chunks = result["child_chunks"]

        # 2. Embed ONLY children (parents are not used for retrieval)
        print(f"\n[embed] Embedding {len(child_chunks)} child chunks...")
        child_texts = [c["content"] for c in child_chunks]
        child_embeddings = asyncio.run(self.embed_batch(child_texts))

        # 3. Store parents with zero vector (they carry full context + images)
        zero_vector = [0.0] * EMBEDDING_DIM
        parent_texts = [c["content"] for c in parent_chunks]
        parent_metas = [c["meta"] for c in parent_chunks]
        parent_ids = [c["parent_id"] for c in parent_chunks]
        parent_embeddings = [zero_vector] * len(parent_chunks)

        print(f"[store] Inserting {len(parent_chunks)} parents (zero vector)...")
        self.vectorstore.add_embeddings(
            texts=parent_texts,
            metadatas=parent_metas,
            ids=parent_ids,
            embeddings=parent_embeddings,
        )

        # 4. Store children with real embeddings
        child_metas = [c["meta"] for c in child_chunks]
        child_ids = [c["child_id"] for c in child_chunks]

        print(f"[store] Inserting {len(child_chunks)} children (real embeddings)...")
        self.vectorstore.add_embeddings(
            texts=child_texts,
            metadatas=child_metas,
            ids=child_ids,
            embeddings=child_embeddings,
        )

        total_stored = len(parent_chunks) + len(child_chunks)
        print(f"[store] Total inserted: {total_stored} chunks "
              f"({len(parent_chunks)} parents + {len(child_chunks)} children)")

        # 5. Index
        self.create_index(lists=100)

        return {
            "status": "completed",
            "document_id": base_meta["document_id"],
            "parent_chunks": len(parent_chunks),
            "child_chunks": len(child_chunks),
            "total_stored": total_stored,
            "total_pages": base_meta["total_pages"],
        }


# ---------------------------------------------------------------------------
# Retrieval Helper (use in your RAG agent's search tools)
# ---------------------------------------------------------------------------
def retrieve_with_parent_context(
    query_embedding: list[float],
    vectorstore: PGVector,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Search CHILD chunks by similarity, then pull their PARENT chunks
    for full context. This is how your RAG agent should retrieve.

    Returns deduplicated parent chunks (with full section context + images).

    Usage in your vector_search_tool / hybrid_search_tool:
        matched_children = vectorstore.similarity_search(query, k=top_k, filter={"is_parent": False})
        parent_ids = list({cm.metadata["parent_chunk_id"] for cm in matched_children})
        parents = vectorstore.get_by_ids(parent_ids)
        return parents  # Full context for LLM
    """
    # Search children only
    child_results = vectorstore.similarity_search_by_vector(
        embedding=query_embedding,
        k=top_k,
        filter={"is_parent": False},
    )

    # Collect unique parent IDs (preserve order, deduplicate)
    seen_parents: OrderedDict[str, None] = OrderedDict()
    for child_doc in child_results:
        pid = child_doc.metadata.get("parent_chunk_id")
        if pid and pid not in seen_parents:
            seen_parents[pid] = None

    if not seen_parents:
        return []

    # Pull parent chunks
    parent_ids = list(seen_parents.keys())
    parent_docs = vectorstore.get_by_ids(parent_ids)

    results = []
    for pdoc in parent_docs:
        results.append({
            "content": pdoc.page_content,
            "metadata": pdoc.metadata,
            "matched_child_count": sum(
                1 for cd in child_results
                if cd.metadata.get("parent_chunk_id") == pdoc.metadata.get("chunk_index")
            ),
        })

    print(
        f"[retrieve] {len(child_results)} child matches -> "
        f"{len(parent_docs)} unique parents returned"
    )
    return results


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Runner (Update this section at the bottom of ingestion1.py)
# ---------------------------------------------------------------------------
def run_ingestion_pipeline(file_path: str, base_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    db_conn = os.getenv("DATABASE_URL")
    if not db_conn:
        raise EnvironmentError("DATABASE_URL environment variable is required.")
        
    # Generate metadata automatically if not provided by the caller
    if base_meta is None:
        doc_name = os.path.basename(file_path)
        base_meta = generate_document_metadata(
            doc_name=doc_name,
            source_file=file_path,
            search_keywords=["Banking", "Product", doc_name.split(".")[0]],
        )
        
    pipeline = FinancialIngestionPipeline(db_connection_string=db_conn)
    return pipeline.process_and_store(file_path, base_meta)

if __name__ == "__main__":
    if not os.path.exists(HARD_CODED_PDF_PATH):
        print(f"File not found: {HARD_CODED_PDF_PATH}")
        exit(1)

    print(f"Starting ingestion: {os.path.basename(HARD_CODED_PDF_PATH)}")
    doc_name = os.path.basename(HARD_CODED_PDF_PATH)
    base_meta = generate_document_metadata(
        doc_name=doc_name,
        source_file=HARD_CODED_PDF_PATH,
        search_keywords=["Financial", "Report", doc_name.split(".")[0]],
    )

    try:
        result = run_ingestion_pipeline(HARD_CODED_PDF_PATH, base_meta)
        print("\n Ingestion completed!")
        print(f"  Document ID   : {result['document_id']}")
        print(f"  Parent Chunks : {result['parent_chunks']} (context, not embedded)")
        print(f"  Child Chunks  : {result['child_chunks']} (retrieval, embedded)")
        print(f"  Total Stored  : {result['total_stored']}")
        print(f"  Pages         : {result['total_pages']}")
    except Exception as e:
        print(f"\n Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
