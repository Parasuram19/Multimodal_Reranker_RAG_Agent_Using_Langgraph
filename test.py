#!/usr/bin/env python3
"""
Generates the requested project structure for Multimodal Agentic RAG.
Run: python setup_project.py
"""

from pathlib import Path

def create_project_structure(base_dir: str = "."):
    # Define all target files exactly as requested
    target_files = [
        "api/v1/agents/rag_answer_agent.py",
        "api/v1/routes/admin.py",
        "api/v1/routes/query.py",
        "api/v1/schemas/query_schema.py",
        "api/v1/services/query_service.py",
        "api/v1/services/upload_service.py",
        "api/v1/tools/fts_search_tool.py",
        "api/v1/tools/hybrid_search_tool.py",
        "api/v1/tools/vector_search_tool.py",
        "api/v1/utils/reranker.py",
        "api/v1/utils/scoring.py",
        "core/helper.py",
        "ingestion/ingestion.py"
    ]

    base = Path(base_dir)

    print("📂 Creating directory structure & files...")
    for rel_path in target_files:
        file_path = base / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if not file_path.exists():
            file_path.touch()
            print(f"✅ Created: {file_path}")
        else:
            print(f"⏭️  Skipped (exists): {file_path}")

    # Add __init__.py for proper Python package recognition
    package_dirs = [
        "api", "api/v1", "api/v1/agents", "api/v1/routes",
        "api/v1/schemas", "api/v1/services", "api/v1/tools",
        "api/v1/utils", "core", "ingestion"
    ]
    
    print("\n📦 Creating package markers (__init__.py)...")
    for dir_name in package_dirs:
        init_file = base / dir_name / "__init__.py"
        if not init_file.exists():
            init_file.touch()
            print(f"✅ Created: {init_file}")

    print("\n🎉 Project structure generated successfully!")
    print("💡 Tip: Run `python -m venv .venv && source .venv/bin/activate` (Linux/macOS) or `.venv\\Scripts\\activate` (Windows)")

if __name__ == "__main__":
    create_project_structure()