"""
Scoring Module — 100% safe against list/dict/None content from Gemini/PGVector
"""
from __future__ import annotations
import math
import re
import json
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as _sklearn_cosine_sim

_SIMPLE_PUNCT_RE = re.compile(r"[^\w\s]")

# 🔑 SAFE TEXT EXTRACTOR (replaces all .strip() vulnerabilities)
def _to_text(content: Any) -> str:
    if content is None: return ""
    if isinstance(content, list):
        return " ".join(_to_text(item) for item in content if item is not None)
    if isinstance(content, dict):
        # Handle JSON table/image chunks from ingestion.py
        if content.get("type") == "table":
            parts = [content.get(k, "") for k in ("section_header", "docling_caption", "llm_generated_caption", "table_data")]
            return " ".join(_to_text(p) for p in parts)
        elif content.get("type") == "image":
            parts = [content.get(k, "") for k in ("section_header", "docling_caption", "llm_generated_caption")]
            return " ".join(_to_text(p) for p in parts)
        return " ".join(_to_text(v) for v in content.values())
    return str(content).strip()

def _normalize(text: str) -> str:
    return _SIMPLE_PUNCT_RE.sub(" ", text.lower()).strip()

def _bm25_scores(query_tokens: List[str], corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    N = len(corpus_tokens)
    if N == 0: return np.array([])
    avgdl = np.mean([len(d) for d in corpus_tokens]) or 1
    idf: Dict[str, float] = {}
    for t in query_tokens:
        df = sum(1 for d in corpus_tokens if t in d)
        idf[t] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
    scores = np.zeros(N, dtype=np.float64)
    for i, doc in enumerate(corpus_tokens):
        dl = len(doc)
        tf_map = {}
        for t in doc: tf_map[t] = tf_map.get(t, 0) + 1
        s = 0.0
        for t in query_tokens:
            if t not in idf: continue
            tf = tf_map.get(t, 0)
            s += idf[t] * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
        scores[i] = s
    mx = scores.max()
    return scores / mx if mx > 0 else scores

def _coverage_ratio(q_tokens: List[str], c_tokens: List[str]) -> float:
    if not q_tokens: return 0.0
    return sum(1 for t in set(q_tokens) if t in set(c_tokens)) / len(set(q_tokens))

def _length_penalty(c_tokens: List[str], ideal_len: int = 120) -> float:
    if not c_tokens: return 0.0
    return math.exp(-0.5 * (abs(len(c_tokens) - ideal_len) / (ideal_len * 0.6)) ** 2)

def compute_cosine_similarity(query: str, chunks: List[Any]) -> List[float]:
    if not chunks or not query: return [0.0] * len(chunks)
    corpus = [_normalize(_to_text(c)) for c in chunks]
    norm_q = _normalize(query)
    valid_idx = [i for i, c in enumerate(corpus) if c]
    if not valid_idx: return [0.0] * len(chunks)
    
    vec = TfidfVectorizer(stop_words="english", min_df=1, norm="l2", sublinear_tf=True)
    try:
        mat = vec.fit_transform([corpus[i] for i in valid_idx] + [norm_q])
        sims = _sklearn_cosine_sim(mat[-1:], mat[:-1]).flatten()
        scores = [0.0] * len(chunks)
        for i, s in zip(valid_idx, sims): scores[i] = float(s)
        return scores
    except ValueError: return [0.0] * len(chunks)

def score_chunks(query: str, chunks: List[Any], weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    if not chunks: return []
    texts = [_to_text(c) for c in chunks]
    cos_sims = compute_cosine_similarity(query, texts)
    q_tokens = _normalize(query).split()
    c_tokens = [_normalize(t).split() for t in texts]
    bm25 = _bm25_scores(q_tokens, c_tokens) if q_tokens else np.zeros(len(chunks))
    
    w = weights or {"cosine_similarity": 0.40, "bm25": 0.35, "coverage_ratio": 0.15, "length_penalty": 0.10}
    out = []
    for i in range(len(chunks)):
        ct = c_tokens[i]
        rel = (w["cosine_similarity"] * cos_sims[i] + 
               w["bm25"] * float(bm25[i]) + 
               w["coverage_ratio"] * _coverage_ratio(q_tokens, ct) + 
               w["length_penalty"] * _length_penalty(ct))
        out.append({
            "cosine_similarity": round(cos_sims[i], 6),
            "bm25": round(float(bm25[i]), 6),
            "relevance_score": round(max(0.0, min(1.0, rel)), 6)
        })
    return out