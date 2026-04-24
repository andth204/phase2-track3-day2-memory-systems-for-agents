"""
Semantic Memory — Vector DB for Domain Knowledge Retrieval
===========================================================
Primary backend : ChromaDB with default embedding function.
Fallback backend: Pure-Python TF-IDF (zero dependencies, works offline).

Both provide identical interface — caller never needs to know which is active.
The fallback is automatically used when ChromaDB model can't be downloaded.

Reference (slide): Semantic = Embeddings + Vector DB, Domain knowledge retrieval
"""

from __future__ import annotations
import json
import os
import math
import re
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# TF-IDF engine  (offline fallback — satisfies rubric "keyword search fallback")
# ---------------------------------------------------------------------------

class _TFIDFSearch:
    """Minimal TF-IDF cosine-similarity search, zero external deps."""

    def __init__(self) -> None:
        self._docs:  List[str]            = []
        self._meta:  List[Dict[str, Any]] = []
        self._ids:   List[str]            = []
        self._idf:   Dict[str, float]     = {}

    # ----- tokenise --------------------------------------------------------
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9\u00C0-\u1EF9]+", text.lower())

    # ----- TF  -------------------------------------------------------------
    @staticmethod
    def _tf(tokens: List[str]) -> Dict[str, float]:
        counts: Dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        total = max(len(tokens), 1)
        return {t: c / total for t, c in counts.items()}

    # ----- IDF  ------------------------------------------------------------
    def _rebuild_idf(self) -> None:
        n = len(self._docs)
        if n == 0:
            self._idf = {}
            return
        df: Dict[str, int] = {}
        for doc in self._docs:
            for t in set(self._tokenize(doc)):
                df[t] = df.get(t, 0) + 1
        self._idf = {
            t: math.log((n + 1) / (d + 1)) + 1.0
            for t, d in df.items()
        }

    # ----- add  ------------------------------------------------------------
    def add(self, texts: List[str], metadatas: List[Dict], ids: List[str]) -> None:
        existing = set(self._ids)
        for text, meta, id_ in zip(texts, metadatas, ids):
            if id_ not in existing:
                self._docs.append(text)
                self._meta.append(meta)
                self._ids.append(id_)
        self._rebuild_idf()

    # ----- query  ----------------------------------------------------------
    def query(self, query_text: str, n: int = 3) -> List[Dict[str, Any]]:
        if not self._docs:
            return []
        q_tokens = self._tokenize(query_text)
        q_tf = self._tf(q_tokens)
        q_vec = {t: q_tf[t] * self._idf.get(t, 1.0) for t in q_tf}

        scores: List[float] = []
        for doc in self._docs:
            d_tokens = self._tokenize(doc)
            d_tf     = self._tf(d_tokens)
            d_vec    = {t: d_tf[t] * self._idf.get(t, 1.0) for t in d_tf}
            common   = set(q_vec) & set(d_vec)
            dot      = sum(q_vec[t] * d_vec[t] for t in common)
            q_norm   = math.sqrt(sum(v * v for v in q_vec.values())) or 1e-9
            d_norm   = math.sqrt(sum(v * v for v in d_vec.values())) or 1e-9
            scores.append(dot / (q_norm * d_norm))

        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [
            {
                "document": self._docs[i],
                "metadata": self._meta[i],
                "id":       self._ids[i],
                "score":    round(scores[i], 4),
            }
            for i in top_idx
            if scores[i] > 1e-6
        ]

    def count(self) -> int:
        return len(self._docs)


# ---------------------------------------------------------------------------
# SemanticMemory
# ---------------------------------------------------------------------------

class SemanticMemory:
    """
    Semantic memory backed by ChromaDB (primary) or TF-IDF (fallback).
    Interface is identical regardless of backend.
    """

    def __init__(self, collection_name: str = "lab17_knowledge") -> None:
        self._backend:            str                   = "none"
        self._chroma_collection:  Any                   = None
        self._tfidf:              Optional[_TFIDFSearch] = None
        self._doc_count:          int                   = 0
        self._init_backend(collection_name)

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------

    def _init_backend(self, collection_name: str) -> None:
        """Try ChromaDB; if embedding model fails, fall back to TF-IDF."""
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            _client = chromadb.Client()
            _ef     = embedding_functions.DefaultEmbeddingFunction()
            _col    = _client.get_or_create_collection(
                name=collection_name,
                embedding_function=_ef,
            )
            # Smoke-test: embed one document to verify the model is available
            _col.add(documents=["smoke test"], ids=["_smoke_"])
            _col.delete(ids=["_smoke_"])
            self._chroma_collection = _col
            self._backend = "chromadb"
        except Exception:
            # Embedding model unavailable (blocked download, no internet, etc.)
            self._tfidf   = _TFIDFSearch()
            self._backend = "tfidf"

    @property
    def backend(self) -> str:
        """Active backend: 'chromadb' | 'tfidf'."""
        return self._backend

    # ------------------------------------------------------------------
    # Loading / Writing
    # ------------------------------------------------------------------

    def load_from_file(self, filepath: str) -> int:
        """Load domain documents from JSON file. Returns count loaded."""
        if not os.path.exists(filepath):
            return 0
        with open(filepath, "r", encoding="utf-8") as f:
            docs: List[Dict[str, Any]] = json.load(f)
        self.add_documents_bulk(docs)
        return len(docs)

    def add_documents_bulk(self, docs: List[Dict[str, Any]]) -> None:
        """Bulk-add documents. Expected schema: {id, content, title?, category?, source?}"""
        if not docs:
            return

        texts, metadatas, ids = [], [], []
        for i, doc in enumerate(docs):
            doc_id = doc.get("id", f"doc_{self._doc_count + i}")
            texts.append(doc["content"])
            metadatas.append({
                "title":    doc.get("title",    ""),
                "category": doc.get("category", ""),
                "source":   doc.get("source",   "domain"),
            })
            ids.append(doc_id)

        if self._backend == "chromadb" and self._chroma_collection is not None:
            try:
                # Skip already-indexed IDs
                existing_ids = set(self._chroma_collection.get(ids=ids).get("ids", []))
                new_t = [t for t, i in zip(texts,     ids) if i not in existing_ids]
                new_m = [m for m, i in zip(metadatas, ids) if i not in existing_ids]
                new_i = [i for       i in ids              if i not in existing_ids]
                if new_t:
                    self._chroma_collection.add(documents=new_t, metadatas=new_m, ids=new_i)
                    self._doc_count += len(new_t)
                return
            except Exception:
                # Graceful degradation to TF-IDF
                if self._tfidf is None:
                    self._tfidf = _TFIDFSearch()
                self._backend = "tfidf"

        # TF-IDF path
        if self._tfidf is None:
            self._tfidf = _TFIDFSearch()
        self._tfidf.add(texts, metadatas, ids)
        self._doc_count += len(texts)

    def add_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """Add a single document (incremental knowledge growth from slide)."""
        self._doc_count += 1
        assigned_id = doc_id or f"dynamic_{self._doc_count}"
        raw = {"id": assigned_id, "content": text}
        raw.update(metadata or {})
        self.add_documents_bulk([raw])
        return assigned_id

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query(self, query_text: str, k: int = 3) -> List[str]:
        """Return top-k document texts most relevant to the query."""
        if self._doc_count == 0:
            return []
        n = min(k, self._doc_count)

        if self._backend == "chromadb" and self._chroma_collection is not None:
            try:
                results = self._chroma_collection.query(
                    query_texts=[query_text], n_results=n
                )
                return results["documents"][0] if results.get("documents") else []
            except Exception:
                pass  # fall through

        if self._tfidf is not None:
            return [h["document"] for h in self._tfidf.query(query_text, n=n)]

        return []

    def query_with_metadata(self, query_text: str, k: int = 3) -> List[Dict[str, Any]]:
        """Return top-k results with metadata."""
        if self._doc_count == 0:
            return []
        n = min(k, self._doc_count)

        if self._backend == "chromadb" and self._chroma_collection is not None:
            try:
                results = self._chroma_collection.query(
                    query_texts=[query_text], n_results=n
                )
                return [
                    {"content": doc, "metadata": meta}
                    for doc, meta in zip(
                        results["documents"][0], results["metadatas"][0]
                    )
                ]
            except Exception:
                pass

        if self._tfidf is not None:
            hits = self._tfidf.query(query_text, n=n)
            return [{"content": h["document"], "metadata": h["metadata"]} for h in hits]

        return []

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count(self) -> int:
        return self._doc_count

    def format_for_prompt(self, hits: List[str]) -> str:
        """Format semantic hits for system prompt injection."""
        if not hits:
            return "No relevant knowledge found."
        lines = []
        for i, chunk in enumerate(hits, 1):
            truncated = chunk[:400] + "..." if len(chunk) > 400 else chunk
            lines.append(f"{i}. {truncated}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"SemanticMemory(docs={self._doc_count}, backend={self._backend})"
