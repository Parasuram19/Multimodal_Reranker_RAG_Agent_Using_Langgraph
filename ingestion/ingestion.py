# """
# Financial Document Ingestion Pipeline (Simplified)
# - Docling for PDF parsing (text, tables, images)
# - Gemini embedding-2-preview (1536-dim vectors)
# - PGVector with IVFFlat index for fast retrieval
# """

# import os
# import uuid
# import asyncio
# import base64
# import io
# import re
# from datetime import datetime, timezone
# from typing import List, Dict, Any, Optional

# from docling.datamodel.base_models import InputFormat
# from docling.datamodel.pipeline_options import PdfPipelineOptions
# from docling.document_converter import DocumentConverter, PdfFormatOption
# from docling_core.types.doc import TableItem, PictureItem, TextItem
# from google.genai import types
# from google import genai
# from langchain_postgres import PGVector
# from sqlalchemy import create_engine, text
# from dotenv import load_dotenv

# load_dotenv(override=True)

# # ---------------------------------------------------------------------------
# # Constants
# # ---------------------------------------------------------------------------
# EMBEDDING_DIM = 1536
# EMBEDDING_MODEL = "gemini-embedding-2-preview"
# CHUNK_SIZE = 1500
# CHUNK_OVERLAP = 300

# # 🔹 Edit these for your environment
# HARD_CODED_PDF_PATH = r"C:\Users\t91-labuser015568\Desktop\TCS_GEN_AI\multimodal-reranker-agentic-rag\data\RIL-Media-Release-RIL-Q2-FY2024-25-mini.pdf"


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

#         # PGVector store (pre-computed embeddings via add_embeddings)
#         self.vectorstore = PGVector(
#             embeddings=None,
#             connection=self.conn_string,
#             collection_name=self.collection_name,
#             use_jsonb=True,
#         )

#         # Gemini client
#         api_key = os.getenv("GEMINI_API_KEY")
#         if not api_key:
#             raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")
#         self.gemini = genai.Client(api_key=api_key)

#     # ---- Embedding --------------------------------------------------------
#     async def embed_batch(self, texts: list[str]) -> list[list[float]]:
#         """Embed a batch of texts using Gemini (1536-dim)."""

#         async def _embed_one(content: str) -> list[float]:
#             try:
#                 # BEFORE (returns 3072-dim)
#                 # resp = self.gemini.models.embed_content(
#                 #     model=EMBEDDING_MODEL, contents=[content]
#                 # )

#                 # AFTER (forced to 1536-dim)
#                 resp = self.gemini.models.embed_content(
#                     model=EMBEDDING_MODEL,
#                     contents=[content],
#                     config=types.EmbedContentConfig(
#                         output_dimensionality=EMBEDDING_DIM   # 1536
#                     ),
#                 )
#                 return resp.embeddings[0].values
#             except Exception as e:
#                 print(f"Embedding failed: {e}")
#                 return [0.0] * EMBEDDING_DIM

#         return await asyncio.gather(*[_embed_one(t) for t in texts])

#     # ---- Table → plain text -----------------------------------------------
#     @staticmethod
#     def table_to_text(node: TableItem, doc) -> str:
#         """Convert a TableItem to 'Header: value' rows."""
#         # Preferred: pandas DataFrame
#         if hasattr(node, "export_to_dataframe"):
#             try:
#                 df = node.export_to_dataframe(doc)
#                 if df is not None and not df.empty:
#                     lines = []
#                     for _, row in df.iterrows():
#                         pairs = [f"{c}: {v}" for c, v in zip(df.columns, row)
#                                  if str(v).strip() not in ("", "nan", "None")]
#                         if pairs:
#                             lines.append("  |  ".join(pairs))
#                     return "\n".join(lines)
#             except Exception:
#                 pass

#         # Fallback: strip HTML tags
#         if hasattr(node, "export_to_html"):
#             try:
#                 html = node.export_to_html(doc) or ""
#                 return re.sub(r"<[^>]+>", " ", html)
#             except Exception:
#                 pass

#         return getattr(node, "text", "").strip()

#     # ---- Image → base64 ---------------------------------------------------
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

#     # ---- IVFFlat index (1536-dim) -----------------------------------------
#     def create_index(self, lists: int = 100) -> None:
#         """Create / re-create IVFFlat index on the embedding column."""
#         engine = create_engine(self.conn_string)
#         with engine.connect() as conn:
#             conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
#             conn.commit()
#             conn.execute(text(
#                 f"ALTER TABLE langchain_pg_embedding "
#                 f"ALTER COLUMN embedding TYPE vector({EMBEDDING_DIM}) "
#                 f"USING embedding::vector({EMBEDDING_DIM})"
#             ))
#             conn.commit()
#             conn.execute(text(
#                 f"CREATE INDEX IF NOT EXISTS ivfflat_policy_idx "
#                 f"ON langchain_pg_embedding "
#                 f"USING ivfflat (embedding vector_cosine_ops) "
#                 f"WITH (lists = {lists})"
#             ))
#             conn.commit()
#             print(f"IVFFlat index ready (lists={lists}, dim={EMBEDDING_DIM})")

#     # ---- Parse PDF --------------------------------------------------------
#     def parse_document(self, file_path: str,
#                        base_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
#         """Parse a PDF into text / table / image chunks."""
#         result = self.converter.convert(file_path)
#         doc = result.document

#         base_meta["total_pages"] = len(doc.pages) if hasattr(doc, "pages") else 0
#         source_file = os.path.basename(file_path)
#         chunks: List[Dict[str, Any]] = []
#         section = None
#         idx = 0

#         for item in doc.iterate_items():
#             node = item[0] if isinstance(item, tuple) else item

#             label = str(getattr(node, "label", "")).lower()
#             if label in ("page_header", "page_footer"):
#                 continue

#             # Page number from provenance
#             prov = getattr(node, "prov", None)
#             page_no = prov[0].page_no + 1 if prov else 1
#             bbox = None
#             if prov and hasattr(prov[0], "bbox") and prov[0].bbox:
#                 b = prov[0].bbox
#                 bbox = [b.l, b.t, b.r, b.b]

#             def _meta(modality: str, elem_type: str, img_b64=None):
#                 m = {**base_meta, "page": page_no, "chunk_index": idx,
#                      "modality": modality, "element_type": elem_type,
#                      "section": section, "source_file": source_file,
#                      "page_number": page_no, "bbox": bbox,
#                      "image_base64": img_b64}
#                 return m

#             # --- Section headers ---
#             if "section_header" in label or label == "title":
#                 text = getattr(node, "text", "").strip()
#                 if text:
#                     section = text
#                     chunks.append({"content": text, "meta": _meta("text", label)})
#                     idx += 1

#             # --- Tables ---
#             elif isinstance(node, TableItem) or "table" in label:
#                 txt = self.table_to_text(node, doc)
#                 if txt:
#                     chunks.append({"content": txt, "meta": _meta("table", "table")})
#                     idx += 1

#             # --- Images ---
#             elif isinstance(node, PictureItem) or "picture" in label or "figure" in label or label == "chart":
#                 img_b64 = self.image_to_base64(node, doc)
#                 caption = (getattr(node, "text", "") or "").strip()
#                 content = caption or f"[Image on page {page_no}]"
#                 chunks.append({"content": content, "meta": _meta("image", "picture", img_b64)})
#                 idx += 1

#             # --- Text / paragraphs / lists ---
#             elif isinstance(node, TextItem) or label in (
#                 "text", "paragraph", "list_item", "caption", "footnote"
#             ):
#                 text = getattr(node, "text", "").strip()
#                 if not text:
#                     continue
#                 sub_chunks = split_text(text) if len(text) > CHUNK_SIZE else [text]
#                 for s in sub_chunks:
#                     chunks.append({"content": s, "meta": _meta("text", label)})
#                     idx += 1

#         return chunks

#     # ---- Main entry -------------------------------------------------------
#     def process_and_store(self, file_path: str,
#                           base_meta: Dict[str, Any]) -> Dict[str, Any]:
#         """Parse → Embed → Store → Index."""
#         chunks = self.parse_document(file_path, base_meta)
#         print(f"Docling produced {len(chunks)} chunks")

#         # Embed all chunks
#         embeddings = asyncio.run(self.embed_batch([c["content"] for c in chunks]))

#         # Store in PGVector
#         texts = [c["content"] for c in chunks]
#         metas = [c["meta"] for c in chunks]
#         ids = [f"{m['document_id']}_{m['chunk_index']}" for m in metas]

#         self.vectorstore.add_embeddings(texts=texts, metadatas=metas,
#                                         ids=ids, embeddings=embeddings)
#         print(f"Inserted {len(chunks)} chunks → '{self.collection_name}'")

#         # Build IVFFlat index
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

"""
Financial Document Ingestion Pipeline
- Docling for PDF parsing (text, tables, images)
- Gemini embedding-2-preview (1536-dim vectors via output_dimensionality)
- LLM-generated captions for tables and images (saved as JSON in content)
- PGVector with IVFFlat index for fast retrieval
"""

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

#         # PGVector store (pre-computed embeddings via add_embeddings)
#         self.vectorstore = PGVector(
#             embeddings=None,
#             connection=self.conn_string,
#             collection_name=self.collection_name,
#             use_jsonb=True,
#         )

#         # Gemini client
#         api_key = os.getenv("GEMINI_API_KEY")
#         if not api_key:
#             raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")
#         self.gemini = genai.Client(api_key=api_key)

#     # ---- Embedding --------------------------------------------------------
#     async def embed_batch(self, texts: list[str]) -> list[list[float]]:
#         """Embed a batch of texts using Gemini (1536-dim)."""

#         async def _embed_one(content: str) -> list[float]:
#             try:
#                 resp = self.gemini.models.embed_content(
#                     model=EMBEDDING_MODEL,
#                     contents=[content],
#                     config=types.EmbedContentConfig(
#                         output_dimensionality=EMBEDDING_DIM
#                     ),
#                 )
#                 return resp.embeddings[0].values
#             except Exception as e:
#                 print(f"Embedding failed: {e}")
#                 return [0.0] * EMBEDDING_DIM

#         return await asyncio.gather(*[_embed_one(t) for t in texts])

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
#         """Prepare DB -> Parse -> LLM Captions -> Embed -> Store -> Index."""
#         # Step 0: Ensure vector column has correct dimensionality BEFORE insert
#         self.prepare_vector_column()

#         # Step 1: Parse PDF + generate LLM captions
#         chunks = self.parse_document(file_path, base_meta)
#         print(f"Docling produced {len(chunks)} chunks")

#         # Step 2: Embed all chunks
#         embeddings = asyncio.run(self.embed_batch([c["content"] for c in chunks]))

#         # Step 3: Store in PGVector
#         texts = [c["content"] for c in chunks]
#         metas = [c["meta"] for c in chunks]
#         ids = [f"{m['document_id']}_{m['chunk_index']}" for m in metas]

#         self.vectorstore.add_embeddings(texts=texts, metadatas=metas,
#                                         ids=ids, embeddings=embeddings)
#         print(f"Inserted {len(chunks)} chunks -> '{self.collection_name}'")

#         # Step 4: Build IVFFlat index (data-safe, no row deletion)
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


"""
Financial Document Ingestion Pipeline
- Docling for PDF parsing (text, tables, images)
- GoogleGenerativeAIEmbeddings (gemini-embedding-2-preview, 1536-dim)
- LLM-generated captions for tables and images (saved as JSON in content)
- PGVector with IVFFlat index for fast retrieval
"""

import os
import uuid
import asyncio
import base64
import io
import json
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import TableItem, PictureItem, TextItem
from google.genai import types
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 1536
EMBEDDING_MODEL = "gemini-embedding-2-preview"
LLM_MODEL = os.getenv("GEMINI_MODEL")
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300

# Edit these for your environment
HARD_CODED_PDF_PATH = r"C:\Users\t91-labuser015568\Desktop\TCS_GEN_AI\multimodal-reranker-agentic-rag\data\RIL-Media-Release-RIL-Q2-FY2024-25-mini.pdf"

TABLE_CAPTION_PROMPT = """\
You are a financial document analyst. Given the following table data extracted from a PDF,
generate a clear, concise caption that describes what the table contains, including key metrics,
time periods, and units if present.

Section: {section}
Table Data:
{table_data}

Respond with ONLY the caption text, no JSON, no quotes. Keep it under 200 characters."""

IMAGE_CAPTION_PROMPT = """\
You are a financial document analyst. Given the section context and the image below,
generate a clear, concise caption describing what the image/chart shows, including
key data points, trends, or labels visible in the image.

Section: {section}
Docling Caption: {docling_caption}

Respond with ONLY the caption text, no JSON, no quotes. Keep it under 200 characters."""


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


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character windows."""
    chunks, start, step = [], 0, size - overlap
    while start < len(text):
        chunks.append(text[start:start + size])
        start += step
    return chunks


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

        # ---- LangChain GoogleGenerativeAIEmbeddings (replaces direct API calls) ----
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=api_key,
        )

        # PGVector store — now receives the embeddings object directly
        self.vectorstore = PGVector(
            embeddings=self.embeddings,
            connection=self.conn_string,
            collection_name=self.collection_name,
            use_jsonb=True,
        )

        # Gemini client (kept only for LLM caption generation)
        self.gemini = genai.Client(api_key=api_key)

    # ---- LLM Caption Generation -------------------------------------------
    async def generate_table_caption(self, table_data: str, section: str) -> str:
        """Call Gemini LLM to generate a caption for table data."""
        try:
            prompt = TABLE_CAPTION_PROMPT.format(
                section=section or "Unknown",
                table_data=table_data[:3000],  # truncate to avoid token limits
            )
            resp = await asyncio.to_thread(
                self.gemini.models.generate_content,
                model=LLM_MODEL,
                contents=prompt,
            )
            return resp.text.strip()
        except Exception as e:
            print(f"[llm] table caption failed: {e}")
            return ""

    async def generate_image_caption(self, section: str, docling_caption: str,
                                     img_base64: str) -> str:
        """Call Gemini LLM (multimodal) to generate a caption for an image."""
        try:
            prompt = IMAGE_CAPTION_PROMPT.format(
                section=section or "Unknown",
                docling_caption=docling_caption or "No caption provided",
            )

            image_part = types.Part.from_bytes(data=base64.b64decode(img_base64), mime_type="image/png")

            resp = await asyncio.to_thread(
                self.gemini.models.generate_content,
                model=LLM_MODEL,
                contents=[prompt, image_part],
            )
            return resp.text.strip()
        except Exception as e:
            print(f"[llm] image caption failed: {e}")
            return ""

    # ---- Table -> plain text -----------------------------------------------
    @staticmethod
    def table_to_text(node: TableItem, doc) -> str:
        """Convert a TableItem to 'Header: value' rows."""
        # Strategy 1 & 2: pandas DataFrame (try with and without doc arg)
        if hasattr(node, "export_to_dataframe"):
            for call_args in [(doc,), ()]:
                try:
                    df = node.export_to_dataframe(*call_args)
                    if df is not None and not df.empty:
                        lines = []
                        headers = [str(c).strip() for c in df.columns]
                        for _, row in df.iterrows():
                            pairs = [
                                f"{h}: {v}" for h, v in zip(headers, row)
                                if str(v).strip() not in ("", "nan", "None")
                            ]
                            if pairs:
                                lines.append("  |  ".join(pairs))
                        if lines:
                            return "\n".join(lines)
                except TypeError:
                    continue
                except Exception as e:
                    print(f"[table] export_to_dataframe failed: {e}")

        # Strategy 3: HTML strip
        if hasattr(node, "export_to_html"):
            try:
                html = node.export_to_html(doc)
                if html:
                    cleaned = re.sub(r"<[^>]+>", " ", html)
                    cleaned = re.sub(r"\s+", " ", cleaned).strip()
                    if cleaned:
                        return cleaned
            except Exception as e:
                print(f"[table] export_to_html failed: {e}")

        # Strategy 4: raw grid data
        grid = getattr(node, "data", None)
        if grid and hasattr(grid, "grid") and grid.grid:
            try:
                lines = []
                for row_cells in grid.grid:
                    cells = [str(c).strip() for c in row_cells
                             if str(c).strip() not in ("", "nan", "None")]
                    if cells:
                        lines.append("  |  ".join(cells))
                if lines:
                    return "\n".join(lines)
            except Exception as e:
                print(f"[table] grid access failed: {e}")

        # Strategy 5: node.text fallback
        fallback = getattr(node, "text", "").strip()
        if fallback:
            return fallback

        print("[table] WARNING: all extraction strategies failed for table node")
        return ""

    # ---- Image -> base64 ---------------------------------------------------
    @staticmethod
    def image_to_base64(node: PictureItem, doc) -> Optional[str]:
        """Extract picture as base64 PNG string."""
        try:
            pil_img = None
            if hasattr(node, "get_image"):
                pil_img = node.get_image(doc)
            elif hasattr(node, "image") and node.image:
                pil_img = getattr(node.image, "pil_image", None)
            if pil_img:
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode()
        except Exception:
            pass
        return None

    # ---- Build JSON content for table/image --------------------------------
    @staticmethod
    def build_table_json(section: str, docling_caption: str,
                         llm_caption: str, table_data: str) -> str:
        """Build JSON content string for a table chunk."""
        obj = {
            "type": "table",
            "section_header": section,
            "docling_caption": docling_caption,
            "llm_generated_caption": llm_caption,
            "table_data": table_data,
        }
        return json.dumps(obj, ensure_ascii=False)

    @staticmethod
    def build_image_json(section: str, docling_caption: str,
                         llm_caption: str, page_no: int) -> str:
        """Build JSON content string for an image chunk."""
        obj = {
            "type": "image",
            "section_header": section,
            "docling_caption": docling_caption,
            "llm_generated_caption": llm_caption,
            "page": page_no,
        }
        return json.dumps(obj, ensure_ascii=False)

    # ---- IVFFlat index (1536-dim) -----------------------------------------
    def prepare_vector_column(self) -> None:
        """Ensure embedding column is typed as vector(EMBEDDING_DIM).

        MUST be called BEFORE inserting any data. If the column has the wrong
        dimensionality (or no type), existing rows are deleted and the column
        is re-typed so new inserts will succeed.
        """
        engine = create_engine(self.conn_string)
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()

            # Check current column dimensionality
            row = conn.execute(text(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = 'langchain_pg_embedding'::regclass "
                "AND attname = 'embedding'"
            )).fetchone()

            current_dim = row[0] if row else -1
            if current_dim != EMBEDDING_DIM:
                conn.execute(text("DELETE FROM langchain_pg_embedding"))
                conn.commit()
                print(f"[db] Cleared existing rows (had dim={current_dim}, need {EMBEDDING_DIM})")

                conn.execute(text(
                    f"ALTER TABLE langchain_pg_embedding "
                    f"ALTER COLUMN embedding TYPE vector({EMBEDDING_DIM}) "
                    f"USING embedding::vector({EMBEDDING_DIM})"
                ))
                conn.commit()
                print(f"[db] Column set to vector({EMBEDDING_DIM})")
            else:
                print(f"[db] Column already vector({EMBEDDING_DIM}) — OK")

    def create_index(self, lists: int = 100) -> None:
        """Create IVFFlat index on the embedding column.

        Call AFTER inserting data. Only creates/recreates the index —
        does NOT touch column type or data.
        """
        engine = create_engine(self.conn_string)
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()

            conn.execute(text("DROP INDEX IF EXISTS ivfflat_policy_idx"))
            conn.commit()

            conn.execute(text(
                f"CREATE INDEX IF NOT EXISTS ivfflat_policy_idx "
                f"ON langchain_pg_embedding "
                f"USING ivfflat (embedding vector_cosine_ops) "
                f"WITH (lists = {lists})"
            ))
            conn.commit()
            print(f"[db] IVFFlat index ready (lists={lists}, dim={EMBEDDING_DIM})")

    # ---- Parse PDF --------------------------------------------------------
    def parse_document(self, file_path: str,
                       base_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse a PDF into text / table / image chunks.

        Tables and images are stored as JSON in the content field with:
        - section_header: the owning section heading
        - docling_caption: caption text extracted by Docling
        - llm_generated_caption: AI-generated caption via Gemini LLM
        - table_data / page: the actual content
        """
        result = self.converter.convert(file_path)
        doc = result.document

        base_meta["total_pages"] = len(doc.pages) if hasattr(doc, "pages") else 0
        source_file = os.path.basename(file_path)
        chunks: List[Dict[str, Any]] = []
        section = None
        last_caption = None
        idx = 0

        # Collect LLM tasks for tables and images (async batch)
        llm_tasks: list = []

        for item in doc.iterate_items():
            node = item[0] if isinstance(item, tuple) else item

            label = str(getattr(node, "label", "")).lower()
            if label in ("page_header", "page_footer"):
                continue

            prov = getattr(node, "prov", None)
            page_no = prov[0].page_no + 1 if prov else 1
            bbox = None
            if prov and hasattr(prov[0], "bbox") and prov[0].bbox:
                b = prov[0].bbox
                bbox = [b.l, b.t, b.r, b.b]

            def _meta(modality: str, elem_type: str, docling_label: str, img_b64=None):
                m = {**base_meta, "page": page_no, "chunk_index": idx,
                     "modality": modality, "element_type": elem_type,
                     "docling_label": docling_label,
                     "section": section, "source_file": source_file,
                     "page_number": page_no, "bbox": bbox,
                     "image_base64": img_b64}
                return m

            # --- Section headers & title ---
            if "section_header" in label or label == "title":
                t = getattr(node, "text", "").strip()
                if t:
                    section = t
                    chunks.append({"content": t, "meta": _meta("text", label, label)})
                    idx += 1

            # --- Caption: remember for next table/image ---
            elif label == "caption":
                t = getattr(node, "text", "").strip()
                if t:
                    last_caption = t
                    chunks.append({"content": t, "meta": _meta("text", "caption", "caption")})
                    idx += 1

            # --- Table: LLM caption + JSON content ---
            elif isinstance(node, TableItem) or "table" in label:
                table_data = self.table_to_text(node, doc)
                if not table_data:
                    print(f"[skip] table on page {page_no} produced no data")
                    continue

                print(f"[table] page {page_no}: {len(table_data)} chars extracted")
                m = _meta("table", "table", "table")
                m["table_title"] = last_caption or section or ""
                m["section_header"] = section
                m["docling_caption"] = last_caption or ""

                # Placeholder chunk — will be filled after LLM batch
                chunks.append({
                    "content": "",  # filled after LLM call
                    "meta": m,
                    "table_data": table_data,
                    "section": section,
                    "docling_caption": last_caption or "",
                    "page_no": page_no,
                })
                llm_tasks.append(("table", len(chunks) - 1))
                idx += 1
                last_caption = None

            # --- Image: LLM caption (multimodal) + JSON content ---
            elif isinstance(node, PictureItem) or "picture" in label or "figure" in label or label == "chart":
                img_b64 = self.image_to_base64(node, doc)
                caption = (getattr(node, "text", "") or "").strip()
                print(f"[image] page {page_no}: extracted"
                      + (f" ({len(img_b64)} bytes base64)" if img_b64 else " (no image data)"))

                m = _meta("image", "picture", "picture", img_b64)
                m["table_title"] = last_caption or section or ""
                m["section_header"] = section
                m["docling_caption"] = caption or last_caption or ""

                chunks.append({
                    "content": "",  # filled after LLM call
                    "meta": m,
                    "img_base64": img_b64,
                    "docling_caption": caption or last_caption or "",
                    "section": section,
                    "page_no": page_no,
                })
                llm_tasks.append(("image", len(chunks) - 1))
                idx += 1
                last_caption = None

            # --- Text / paragraph / list_item / footnote ---
            elif isinstance(node, TextItem) or label in (
                "text", "paragraph", "list_item", "footnote"
            ):
                t = getattr(node, "text", "").strip()
                if not t:
                    continue
                sub_chunks = split_text(t) if len(t) > CHUNK_SIZE else [t]
                for s in sub_chunks:
                    chunks.append({"content": s, "meta": _meta("text", label, label)})
                    idx += 1

        # --- Run all LLM caption tasks in parallel ---
        if llm_tasks:
            print(f"[llm] generating captions for {len(llm_tasks)} tables/images...")
            captions = self._run_llm_captions(chunks, llm_tasks)
            for task_idx, (kind, chunk_idx) in enumerate(llm_tasks):
                chunk = chunks[chunk_idx]
                if kind == "table":
                    chunk["content"] = self.build_table_json(
                        section=chunk["section"],
                        docling_caption=chunk["docling_caption"],
                        llm_caption=captions[task_idx],
                        table_data=chunk["table_data"],
                    )
                    print(f"[llm] table caption: {captions[task_idx][:80]}...")
                elif kind == "image":
                    chunk["content"] = self.build_image_json(
                        section=chunk["section"],
                        docling_caption=chunk["docling_caption"],
                        llm_caption=captions[task_idx],
                        page_no=chunk["page_no"],
                    )
                    print(f"[llm] image caption: {captions[task_idx][:80]}...")

        return chunks

    def _run_llm_captions(self, chunks: list, tasks: list) -> list[str]:
        """Run all LLM caption generation tasks asynchronously in parallel."""

        async def _run():
            async def _caption_for(task_idx: int, kind: str, chunk_idx: int):
                chunk = chunks[chunk_idx]
                if kind == "table":
                    return await self.generate_table_caption(
                        table_data=chunk["table_data"],
                        section=chunk["section"],
                    )
                elif kind == "image":
                    if chunk.get("img_base64"):
                        return await self.generate_image_caption(
                            section=chunk["section"],
                            docling_caption=chunk["docling_caption"],
                            img_base64=chunk["img_base64"],
                        )
                    else:
                        return chunk["docling_caption"] or ""
                return ""

            results = await asyncio.gather(
                *[_caption_for(i, kind, idx) for i, (kind, idx) in enumerate(tasks)]
            )
            return results

        return asyncio.run(_run())

    # ---- Main entry -------------------------------------------------------
    def process_and_store(self, file_path: str,
                          base_meta: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare DB -> Parse -> LLM Captions -> Embed (via LangChain) -> Store -> Index."""
        # Step 0: Ensure vector column has correct dimensionality BEFORE insert
        self.prepare_vector_column()

        # Step 1: Parse PDF + generate LLM captions
        chunks = self.parse_document(file_path, base_meta)
        print(f"Docling produced {len(chunks)} chunks")

        # Step 2: Embed + Store in PGVector using GoogleGenerativeAIEmbeddings
        #         add_texts() calls embed_documents() internally
        texts = [c["content"] for c in chunks]
        metas = [c["meta"] for c in chunks]
        ids = [f"{m['document_id']}_{m['chunk_index']}" for m in metas]

        self.vectorstore.add_texts(texts=texts, metadatas=metas, ids=ids)
        print(f"Inserted {len(chunks)} chunks -> '{self.collection_name}' "
              f"(embedded via GoogleGenerativeAIEmbeddings)")

        # Step 3: Build IVFFlat index (data-safe, no row deletion)
        self.create_index(lists=100)

        return {
            "status": "completed",
            "document_id": base_meta["document_id"],
            "indexed_chunks": len(chunks),
            "total_pages": base_meta["total_pages"],
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_ingestion_pipeline(file_path: str, base_meta: Dict[str, Any]) -> Dict[str, Any]:
    db_conn = os.getenv("DATABASE_URL")
    if not db_conn:
        raise EnvironmentError("DATABASE_URL environment variable is required.")
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
        print(f"Ingestion completed!")
        print(f"  Document ID : {result['document_id']}")
        print(f"  Chunks      : {result['indexed_chunks']}")
        print(f"  Pages       : {result['total_pages']}")
    except Exception as e:
        print(f"Ingestion failed: {e}")
        import traceback
        traceback.print_exc()