from typing import Any
from api.v1.agents.rag_answer_agent import run_rag_agent


def query_documents(
    query: str,

):
    return run_rag_agent(query)