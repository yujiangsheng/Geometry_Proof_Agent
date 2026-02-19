"""rag.py – Retrieval-Augmented Generation for geometry proof assistance.

When the local symbolic engine or LLM cannot solve a problem, this module
provides two external knowledge retrieval mechanisms:

1. **Local RAG** (``LocalRetriever``):
   Maintains a vector store of geometry theorems, lemmas, proof strategies,
   and Lean4 axiom documentation.  Uses Ollama's embedding API (or a
   lightweight fallback TF-IDF) for similarity search.

2. **Web Search** (``WebSearchProvider``):
   Searches the internet for relevant geometry references via pluggable
   backends:
     • DuckDuckGo HTML search (zero-config, no API key)
     • SerpAPI   (``SERPAPI_KEY`` env var)
     • Bing      (``BING_SEARCH_KEY`` env var)

3. **GeometryRAG** (orchestrator):
   Tries local retrieval first → web search fallback → formats context
   for LLM prompt augmentation.

Environment variables
---------------------
SERPAPI_KEY        – API key for SerpAPI (optional)
BING_SEARCH_KEY   – API key for Bing Web Search v7 (optional)
OLLAMA_BASE_URL   – Ollama API base URL (default: http://localhost:11434)
RAG_DATA_DIR      – Directory for local vector store
                    (default: <project>/data/rag)
RAG_EMBED_MODEL   – Ollama embedding model name
                    (default: nomic-embed-text)

Usage::

    from geometry_agent.rag import get_rag

    rag = get_rag()
    ctx = rag.retrieve("inscribed angle theorem cyclic quadrilateral")
    # ctx.local_hits   – ranked docs from local store
    # ctx.web_hits     – ranked snippets from web search
    # ctx.augmented_prompt(question) – ready-to-use LLM prompt

Author:  Jiangsheng Yu
License: MIT
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_RAG_DIR = Path(os.environ.get(
    "RAG_DATA_DIR", str(_PROJECT_ROOT / "data" / "rag"),
))
_OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
_EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "nomic-embed-text")

# Web search API keys (all optional)
_SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "")
_BING_KEY = os.environ.get("BING_SEARCH_KEY", "")


# ═══════════════════════════════════════════════════════════════════════
#  Data types
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Document:
    """A unit of knowledge in the local vector store."""
    doc_id: str
    title: str
    content: str
    source: str = ""          # e.g. "builtin", "mathlib", "user"
    tags: Tuple[str, ...] = ()
    embedding: Optional[List[float]] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        d: Dict[str, Any] = {
            "doc_id": self.doc_id,
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "tags": list(self.tags),
        }
        if self.embedding is not None:
            d["embedding"] = self.embedding
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        return cls(
            doc_id=d["doc_id"],
            title=d["title"],
            content=d["content"],
            source=d.get("source", ""),
            tags=tuple(d.get("tags", ())),
            embedding=d.get("embedding"),
        )


@dataclass
class SearchHit:
    """A single search result (local or web)."""
    title: str
    snippet: str
    score: float = 0.0
    url: str = ""
    source: str = ""          # "local" | "duckduckgo" | "serpapi" | "bing"


@dataclass
class RetrievalContext:
    """Combined retrieval results from local + web sources."""
    query: str
    local_hits: List[SearchHit] = field(default_factory=list)
    web_hits: List[SearchHit] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def has_results(self) -> bool:
        return bool(self.local_hits or self.web_hits)

    @property
    def all_hits(self) -> List[SearchHit]:
        """All hits, local first, then web."""
        return self.local_hits + self.web_hits

    @property
    def best_snippets(self) -> List[str]:
        """Top snippets for LLM context injection (max 5)."""
        return [h.snippet for h in self.all_hits[:5]]

    def augmented_prompt(
        self,
        question: str,
        *,
        max_context_chars: int = 3000,
        lang: str = "zh",
    ) -> str:
        """Build an LLM prompt augmented with retrieval context.

        Parameters
        ----------
        question : str
            The original question or problem description.
        max_context_chars : int
            Truncate context to this length.
        lang : str
            Language for framing text ("zh" or "en").

        Returns
        -------
        str
            A prompt with relevant context prepended.
        """
        if not self.has_results:
            return question

        parts: List[str] = []
        char_count = 0
        for hit in self.all_hits:
            text = f"[{hit.source}] {hit.title}\n{hit.snippet}"
            if char_count + len(text) > max_context_chars:
                break
            parts.append(text)
            char_count += len(text)

        context_block = "\n\n".join(parts)

        if lang == "zh":
            header = (
                "以下是检索到的相关几何知识，请参考后回答问题。\n"
                "如果检索结果不相关，请忽略它们。\n\n"
            )
        else:
            header = (
                "Below is relevant geometry knowledge retrieved for reference.\n"
                "If the retrieved results are not relevant, ignore them.\n\n"
            )

        return f"{header}--- 检索结果 / Retrieved Context ---\n{context_block}\n\n--- 问题 / Question ---\n{question}"


# ═══════════════════════════════════════════════════════════════════════
#  Embedding helpers
# ═══════════════════════════════════════════════════════════════════════

def _ollama_embed(texts: List[str], model: str = _EMBED_MODEL) -> Optional[List[List[float]]]:
    """Get embeddings via Ollama /api/embed endpoint.

    Returns None if Ollama is unreachable or the model is unavailable.
    """
    url = f"{_OLLAMA_BASE}/api/embed"
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    req = Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        embeddings = data.get("embeddings")
        if embeddings and len(embeddings) == len(texts):
            return embeddings
        return None
    except (URLError, OSError, json.JSONDecodeError, KeyError) as exc:
        logger.debug("Ollama embed unavailable: %s", exc)
        return None


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


# ── TF-IDF fallback (when no embedding model is available) ───────────

def _tokenize(text: str) -> List[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric."""
    return re.findall(r"[a-z0-9\u4e00-\u9fff]+", text.lower())


class _TfIdfIndex:
    """Minimal TF-IDF index as fallback when embeddings are unavailable."""

    def __init__(self) -> None:
        self._docs: List[Tuple[str, List[str]]] = []  # (doc_id, tokens)
        self._idf: Dict[str, float] = {}
        self._dirty = True

    def add(self, doc_id: str, text: str) -> None:
        self._docs.append((doc_id, _tokenize(text)))
        self._dirty = True

    def _rebuild_idf(self) -> None:
        if not self._dirty:
            return
        n = len(self._docs)
        if n == 0:
            return
        df: Counter = Counter()
        for _, tokens in self._docs:
            df.update(set(tokens))
        self._idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self._dirty = False

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Return (doc_id, score) pairs ranked by TF-IDF cosine sim."""
        self._rebuild_idf()
        q_tokens = _tokenize(query)
        if not q_tokens or not self._docs:
            return []

        q_tf: Counter = Counter(q_tokens)
        q_vec = {t: q_tf[t] * self._idf.get(t, 0.0) for t in q_tf}
        q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
        if q_norm < 1e-12:
            return []

        results: List[Tuple[str, float]] = []
        for doc_id, tokens in self._docs:
            d_tf: Counter = Counter(tokens)
            d_vec = {t: d_tf[t] * self._idf.get(t, 0.0) for t in d_tf}
            d_norm = math.sqrt(sum(v * v for v in d_vec.values()))
            if d_norm < 1e-12:
                continue
            dot = sum(q_vec.get(t, 0.0) * d_vec.get(t, 0.0) for t in d_vec)
            sim = dot / (q_norm * d_norm)
            results.append((doc_id, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# ═══════════════════════════════════════════════════════════════════════
#  Web search providers
# ═══════════════════════════════════════════════════════════════════════

class WebSearchProvider:
    """Search the web for geometry-related information.

    Tries backends in order of preference:
      1. SerpAPI  (if SERPAPI_KEY is set)
      2. Bing     (if BING_SEARCH_KEY is set)
      3. DuckDuckGo HTML scraping (always available, no key needed)
    """

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        timeout: int = 10,
    ) -> List[SearchHit]:
        """Search the web and return ranked results.

        Parameters
        ----------
        query : str
            Search query (will be augmented with "geometry theorem").
        max_results : int
            Maximum number of results to return.
        timeout : int
            HTTP request timeout in seconds.

        Returns
        -------
        list[SearchHit]
        """
        geo_query = f"{query} geometry theorem proof"

        # Try in order of quality
        if _SERPAPI_KEY:
            hits = self._serpapi(geo_query, max_results, timeout)
            if hits:
                return hits

        if _BING_KEY:
            hits = self._bing(geo_query, max_results, timeout)
            if hits:
                return hits

        return self._duckduckgo(geo_query, max_results, timeout)

    # ── SerpAPI ──────────────────────────────────────────────

    @staticmethod
    def _serpapi(query: str, max_results: int, timeout: int) -> List[SearchHit]:
        params = urlencode({
            "q": query,
            "api_key": _SERPAPI_KEY,
            "engine": "google",
            "num": max_results,
        })
        url = f"https://serpapi.com/search.json?{params}"
        try:
            req = Request(url, method="GET")
            with urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            hits: List[SearchHit] = []
            for r in data.get("organic_results", [])[:max_results]:
                hits.append(SearchHit(
                    title=r.get("title", ""),
                    snippet=r.get("snippet", ""),
                    score=1.0 - len(hits) * 0.1,
                    url=r.get("link", ""),
                    source="serpapi",
                ))
            logger.debug("SerpAPI returned %d results", len(hits))
            return hits
        except (URLError, OSError, json.JSONDecodeError) as exc:
            logger.warning("SerpAPI search failed: %s", exc)
            return []

    # ── Bing Web Search v7 ───────────────────────────────────

    @staticmethod
    def _bing(query: str, max_results: int, timeout: int) -> List[SearchHit]:
        url = f"https://api.bing.microsoft.com/v7.0/search?q={quote_plus(query)}&count={max_results}"
        req = Request(url, method="GET")
        req.add_header("Ocp-Apim-Subscription-Key", _BING_KEY)
        try:
            with urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            hits: List[SearchHit] = []
            for r in data.get("webPages", {}).get("value", [])[:max_results]:
                hits.append(SearchHit(
                    title=r.get("name", ""),
                    snippet=r.get("snippet", ""),
                    score=1.0 - len(hits) * 0.1,
                    url=r.get("url", ""),
                    source="bing",
                ))
            logger.debug("Bing returned %d results", len(hits))
            return hits
        except (URLError, OSError, json.JSONDecodeError) as exc:
            logger.warning("Bing search failed: %s", exc)
            return []

    # ── DuckDuckGo (HTML scraping, no API key) ───────────────

    @staticmethod
    def _duckduckgo(query: str, max_results: int, timeout: int) -> List[SearchHit]:
        """DuckDuckGo Lite HTML search — zero-config fallback."""
        url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"
        req = Request(url, method="GET")
        req.add_header("User-Agent", "Mozilla/5.0 (compatible; GeometryProofAgent/1.0)")
        try:
            with urlopen(req, timeout=timeout) as resp:
                html = resp.read().decode("utf-8", errors="replace")

            # Parse results from DDG Lite HTML
            hits: List[SearchHit] = []
            # DDG Lite wraps each result in <a class="result-link">...</a>
            # followed by <td class="result-snippet">...</td>
            link_pattern = re.compile(
                r'<a[^>]+class="result-link"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
                re.DOTALL,
            )
            snippet_pattern = re.compile(
                r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>',
                re.DOTALL,
            )

            links = link_pattern.findall(html)
            snippets = snippet_pattern.findall(html)

            for i, (href, title_html) in enumerate(links[:max_results]):
                title = re.sub(r"<[^>]*>", "", title_html).strip()
                snippet = ""
                if i < len(snippets):
                    snippet = re.sub(r"<[^>]*>", "", snippets[i]).strip()
                hits.append(SearchHit(
                    title=title,
                    snippet=snippet,
                    score=1.0 - i * 0.1,
                    url=href,
                    source="duckduckgo",
                ))

            logger.debug("DuckDuckGo returned %d results", len(hits))
            return hits
        except (URLError, OSError) as exc:
            logger.warning("DuckDuckGo search failed: %s", exc)
            return []


# ═══════════════════════════════════════════════════════════════════════
#  Local RAG retriever (vector store + TF-IDF fallback)
# ═══════════════════════════════════════════════════════════════════════

class LocalRetriever:
    """Local vector store with Ollama embeddings or TF-IDF fallback.

    Data is persisted in ``data/rag/documents.jsonl``.  On first use,
    the store is seeded with built-in geometry knowledge (see
    ``_BUILTIN_CORPUS``).

    Parameters
    ----------
    data_dir : Path, optional
        Directory for persistence.  Default: ``data/rag/``.
    embed_model : str, optional
        Ollama embedding model name.  Default: ``nomic-embed-text``.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        embed_model: str = _EMBED_MODEL,
    ) -> None:
        self._dir = Path(data_dir) if data_dir else _DEFAULT_RAG_DIR
        self._store_path = self._dir / "documents.jsonl"
        self._embed_model = embed_model
        self._docs: Dict[str, Document] = {}
        self._tfidf = _TfIdfIndex()
        self._use_embeddings = False
        self._lock = threading.Lock()
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy-load: read from disk + seed builtins on first access."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._dir.mkdir(parents=True, exist_ok=True)
            if self._store_path.exists():
                self._load_from_disk()
            # Seed builtins if store is empty
            if not self._docs:
                self._seed_builtins()
                self._save_to_disk()
            # Detect embedding capability
            test = _ollama_embed(["test"], self._embed_model)
            self._use_embeddings = test is not None
            if self._use_embeddings:
                logger.info("RAG: using Ollama embeddings (%s)", self._embed_model)
                self._ensure_embeddings()
            else:
                logger.info("RAG: using TF-IDF fallback (no embedding model)")
            self._loaded = True

    def _load_from_disk(self) -> None:
        with open(self._store_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = Document.from_dict(json.loads(line))
                self._docs[doc.doc_id] = doc
                self._tfidf.add(doc.doc_id, f"{doc.title} {doc.content}")
        logger.debug("RAG: loaded %d docs from disk", len(self._docs))

    def _save_to_disk(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        with open(self._store_path, "w", encoding="utf-8") as f:
            for doc in self._docs.values():
                f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
        logger.debug("RAG: saved %d docs to disk", len(self._docs))

    def _ensure_embeddings(self) -> None:
        """Compute embeddings for any docs that don't have them yet."""
        need_embed = [d for d in self._docs.values() if d.embedding is None]
        if not need_embed:
            return
        # Batch embed (up to 64 at a time)
        batch_size = 64
        for i in range(0, len(need_embed), batch_size):
            batch = need_embed[i : i + batch_size]
            texts = [f"{d.title}. {d.content}" for d in batch]
            embeddings = _ollama_embed(texts, self._embed_model)
            if embeddings:
                for doc, emb in zip(batch, embeddings):
                    doc.embedding = emb
            else:
                logger.warning("RAG: embedding batch %d failed, falling back to TF-IDF", i)
                self._use_embeddings = False
                return
        self._save_to_disk()

    def add_document(self, doc: Document) -> None:
        """Add a document to the store (persisted)."""
        self._ensure_loaded()
        with self._lock:
            if doc.doc_id in self._docs:
                return
            # Compute embedding if available
            if self._use_embeddings and doc.embedding is None:
                embs = _ollama_embed(
                    [f"{doc.title}. {doc.content}"], self._embed_model,
                )
                if embs:
                    doc.embedding = embs[0]
            self._docs[doc.doc_id] = doc
            self._tfidf.add(doc.doc_id, f"{doc.title} {doc.content}")
            # Append to disk
            self._dir.mkdir(parents=True, exist_ok=True)
            with open(self._store_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")

    def search(self, query: str, top_k: int = 5) -> List[SearchHit]:
        """Search the local store for relevant documents."""
        self._ensure_loaded()

        if self._use_embeddings:
            return self._search_embeddings(query, top_k)
        else:
            return self._search_tfidf(query, top_k)

    def _search_embeddings(self, query: str, top_k: int) -> List[SearchHit]:
        embs = _ollama_embed([query], self._embed_model)
        if not embs:
            return self._search_tfidf(query, top_k)

        q_emb = embs[0]
        scored: List[Tuple[float, Document]] = []
        for doc in self._docs.values():
            if doc.embedding is None:
                continue
            sim = _cosine_similarity(q_emb, doc.embedding)
            scored.append((sim, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            SearchHit(
                title=doc.title,
                snippet=doc.content[:500],
                score=sim,
                source="local",
            )
            for sim, doc in scored[:top_k]
        ]

    def _search_tfidf(self, query: str, top_k: int) -> List[SearchHit]:
        results = self._tfidf.search(query, top_k)
        hits: List[SearchHit] = []
        for doc_id, score in results:
            doc = self._docs.get(doc_id)
            if doc:
                hits.append(SearchHit(
                    title=doc.title,
                    snippet=doc.content[:500],
                    score=score,
                    source="local",
                ))
        return hits

    @property
    def doc_count(self) -> int:
        self._ensure_loaded()
        return len(self._docs)

    def _seed_builtins(self) -> None:
        """Populate the store with built-in geometry knowledge."""
        for entry in _BUILTIN_CORPUS:
            doc_id = hashlib.md5(
                entry["title"].encode("utf-8")
            ).hexdigest()[:12]
            doc = Document(
                doc_id=doc_id,
                title=entry["title"],
                content=entry["content"],
                source="builtin",
                tags=tuple(entry.get("tags", ())),
            )
            self._docs[doc_id] = doc
            self._tfidf.add(doc_id, f"{doc.title} {doc.content}")


# ═══════════════════════════════════════════════════════════════════════
#  GeometryRAG – Orchestrator
# ═══════════════════════════════════════════════════════════════════════

class GeometryRAG:
    """Unified retrieval-augmented generation interface.

    Combines local vector retrieval with web search fallback:
      1. Query the local store (fast, offline)
      2. If local results are insufficient (below threshold),
         fall back to web search
      3. Merge, rank, and return ``RetrievalContext``

    Parameters
    ----------
    local_retriever : LocalRetriever, optional
    web_provider : WebSearchProvider, optional
    local_threshold : float
        Minimum local score to skip web search (default 0.4).
    enable_web : bool
        Whether to enable web fallback (default True).
    """

    def __init__(
        self,
        local_retriever: Optional[LocalRetriever] = None,
        web_provider: Optional[WebSearchProvider] = None,
        *,
        local_threshold: float = 0.4,
        enable_web: bool = True,
    ) -> None:
        self._local = local_retriever or LocalRetriever()
        self._web = web_provider or WebSearchProvider()
        self._threshold = local_threshold
        self._enable_web = enable_web

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        force_web: bool = False,
    ) -> RetrievalContext:
        """Retrieve relevant geometry knowledge.

        Parameters
        ----------
        query : str
            Natural-language query about a geometry concept or theorem.
        top_k : int
            Max results from each source.
        force_web : bool
            Always perform web search regardless of local results.

        Returns
        -------
        RetrievalContext
        """
        t0 = time.time()
        ctx = RetrievalContext(query=query)

        # Step 1: Local retrieval
        try:
            ctx.local_hits = self._local.search(query, top_k=top_k)
        except Exception as exc:
            logger.warning("Local RAG search failed: %s", exc)

        # Step 2: Check if web search is needed
        best_local = max((h.score for h in ctx.local_hits), default=0.0)
        need_web = (
            force_web
            or best_local < self._threshold
            or len(ctx.local_hits) < 2
        )

        if need_web and self._enable_web:
            try:
                ctx.web_hits = self._web.search(query, max_results=top_k)
            except Exception as exc:
                logger.warning("Web search failed: %s", exc)

        ctx.elapsed_ms = (time.time() - t0) * 1000
        logger.info(
            "RAG: query=%r  local=%d (best=%.2f)  web=%d  %.0fms",
            query[:60], len(ctx.local_hits), best_local,
            len(ctx.web_hits), ctx.elapsed_ms,
        )
        return ctx

    def retrieve_for_failure(
        self,
        goal_predicate: str,
        goal_args: Sequence[str],
        assumptions_preds: Sequence[str],
        diagnosis: str,
    ) -> RetrievalContext:
        """Retrieve knowledge relevant to a proof search failure.

        Constructs a targeted query from the proof context.

        Parameters
        ----------
        goal_predicate : str
            The predicate of the unproven goal (e.g. "Parallel").
        goal_args : Sequence[str]
            Arguments of the goal fact.
        assumptions_preds : Sequence[str]
            Predicates present in the assumptions.
        diagnosis : str
            The failure diagnosis from CriticReflectAgent.

        Returns
        -------
        RetrievalContext
        """
        # Build a rich query from the proof context
        pred_names = " ".join(sorted(set(assumptions_preds)))
        query = (
            f"prove {goal_predicate} from {pred_names} "
            f"geometry theorem lemma {diagnosis}"
        )
        return self.retrieve(query, force_web=True)

    def add_knowledge(
        self,
        title: str,
        content: str,
        source: str = "user",
        tags: Sequence[str] = (),
    ) -> None:
        """Add new knowledge to the local store.

        Parameters
        ----------
        title : str
            Short title for the knowledge entry.
        content : str
            Full text content.
        source : str
            Provenance (e.g. "user", "mathlib", "web").
        tags : Sequence[str]
            Optional category tags.
        """
        doc_id = hashlib.md5(
            f"{title}:{content[:100]}".encode("utf-8")
        ).hexdigest()[:12]
        doc = Document(
            doc_id=doc_id,
            title=title,
            content=content,
            source=source,
            tags=tuple(tags),
        )
        self._local.add_document(doc)


# ═══════════════════════════════════════════════════════════════════════
#  Built-in geometry knowledge corpus
# ═══════════════════════════════════════════════════════════════════════

_BUILTIN_CORPUS: List[Dict[str, Any]] = [
    # ── Basic parallel / perpendicular ───────────────────────
    {
        "title": "平行线性质 / Parallel Line Properties",
        "content": (
            "平行线的判定与性质：\n"
            "1. 平行线的对称性：若 AB∥CD，则 CD∥AB\n"
            "2. 平行线的传递性：若 AB∥CD 且 CD∥EF，则 AB∥EF\n"
            "3. 平行线与垂直的传递：若 AB∥CD 且 AB⊥EF，则 CD⊥EF\n"
            "4. 内错角定理：平行线被截时，内错角相等\n"
            "5. 同位角定理：平行线被截时，同位角相等\n"
            "Lean4: Parallel, Perpendicular predicates with symmetry/transitivity axioms."
        ),
        "tags": ["parallel", "perpendicular", "basic"],
    },
    {
        "title": "垂直平分线定理 / Perpendicular Bisector Theorem",
        "content": (
            "垂直平分线定理：若 M 是 AB 的中点，且 PM⊥AB，则 PA=PB。\n"
            "逆定理：若 PA=PB，则 P 在 AB 的垂直平分线上。\n"
            "Lean4: perp_bisector_cong / cong_perp_bisector axioms.\n"
            "应用：证明等腰三角形、菱形、等距点轨迹。"
        ),
        "tags": ["perpendicular_bisector", "congruence", "midpoint"],
    },
    # ── Triangles ────────────────────────────────────────────
    {
        "title": "三角形中位线定理 / Midsegment Theorem",
        "content": (
            "中位线定理：三角形一边的两个中点的连线平行于第三边，且等于第三边的一半。\n"
            "设 D 是 AB 中点，E 是 AC 中点，则 DE∥BC 且 DE=BC/2。\n"
            "推论：中位线还导出相似三角形 △ADE ∼ △ABC。\n"
            "Lean4: midsegment_parallel, midsegment_sim_tri axioms.\n"
            "扩展：可以推出平行四边形对角线互相平分。"
        ),
        "tags": ["triangle", "midpoint", "parallel", "similar"],
    },
    {
        "title": "等腰三角形底角定理 / Isosceles Base Angle Theorem",
        "content": (
            "等腰三角形底角相等：若 AB=AC，则 ∠ABC=∠ACB。\n"
            "逆定理：若 ∠ABC=∠ACB，则 AB=AC（等角对等边）。\n"
            "证明方法：作底边中线或底边高，利用 SAS 全等。\n"
            "Lean4: isosceles_base_angle axiom."
        ),
        "tags": ["triangle", "isosceles", "congruence", "angle"],
    },
    {
        "title": "全等三角形 / Congruent Triangles",
        "content": (
            "全等三角形的判定条件：SSS, SAS, ASA, AAS, HL。\n"
            "全等三角形的性质：对应边相等，对应角相等，面积相等。\n"
            "在本系统中 CongTri(A,B,C,D,E,F) 表示 △ABC≅△DEF。\n"
            "规则：congtri_side (推出对应边全等), congtri_angle (推出对应角相等),\n"
            "congtri_eqarea (推出面积相等)。\n"
            "从相似+对应边全等可得全等：congtri_from_sim_cong。"
        ),
        "tags": ["triangle", "congruent", "congtri"],
    },
    {
        "title": "相似三角形 / Similar Triangles",
        "content": (
            "相似三角形的判定：AA, SAS相似, SSS相似。\n"
            "性质：对应角相等，对应边成比例。\n"
            "SimTri(A,B,C,D,E,F) 表示 △ABC∼△DEF。\n"
            "规则：sim_tri_angle (对应角相等), eqratio_from_simtri (对应边成比例),\n"
            "sim_tri_cong (相似+一组对应边相等 → 全等)。\n"
            "中位线与相似：中位线可导出 midsegment_sim_tri。"
        ),
        "tags": ["triangle", "similar", "simtri", "ratio"],
    },
    # ── Circles ──────────────────────────────────────────────
    {
        "title": "圆周角定理 / Inscribed Angle Theorem",
        "content": (
            "圆周角定理：同弧上的圆周角相等。\n"
            "若 A, B, C, D 共圆，则 ∠ACB=∠ADB（当 C, D 在同侧时）。\n"
            "推论：直径所对的圆周角是直角。\n"
            "Lean4: cyclic_inscribed_angle axiom.\n"
            "应用：证明四点共圆、角度关系。"
        ),
        "tags": ["circle", "cyclic", "inscribed_angle"],
    },
    {
        "title": "切线与半径 / Tangent and Radius",
        "content": (
            "切线性质：切线在切点处垂直于半径。\n"
            "Tangent(P, O, r) 表示以 P 为切点的切线。\n"
            "规则：tangent_perp_radius (切线⊥半径), tangent_oncircle (切点在圆上)。\n"
            "应用：切线长定理，切割线定理。"
        ),
        "tags": ["circle", "tangent", "perpendicular"],
    },
    {
        "title": "外心 / Circumcenter",
        "content": (
            "三角形外心是三边垂直平分线的交点，到三个顶点等距。\n"
            "Circumcenter(O, A, B, C) 表示 O 是 △ABC 的外心。\n"
            "规则：circumcenter_cong_ab (OA=OB), circumcenter_cong_bc (OB=OC),\n"
            "circumcenter_oncircle (A, B, C 在以 O 为圆心的圆上)。\n"
            "锐角三角形的外心在三角形内部，钝角三角形在外部，直角三角形在斜边中点。"
        ),
        "tags": ["circle", "circumcenter", "triangle"],
    },
    # ── Advanced: ratio, harmonic, projective ────────────────
    {
        "title": "角平分线定理 / Angle Bisector Theorem",
        "content": (
            "角平分线定理：三角形的角平分线将对边按邻边之比分成两段。\n"
            "AngleBisect(P, A, B, C) 表示 AP 平分 ∠BAC。\n"
            "规则：angle_bisect_eqangle (等角), angle_bisect_eqratio (等比)。\n"
            "内角平分线定理 + 外角平分线定理构成调和分割。"
        ),
        "tags": ["angle_bisector", "ratio", "triangle"],
    },
    {
        "title": "调和点列 / Harmonic Range",
        "content": (
            "调和点列：四点 A, B, C, D 共线且交比 (A,B;C,D)=-1。\n"
            "Harmonic(A, B, C, D) 表示 ABCD 构成调和点列。\n"
            "性质：harmonic_swap (可以交换 A↔B 或 C↔D),\n"
            "harmonic_collinear (四点共线)。\n"
            "与极点极线的关系：极线上的截线形成调和点列。\n"
            "应用：射影几何、交比计算、极点极线。"
        ),
        "tags": ["harmonic", "projective", "cross_ratio"],
    },
    {
        "title": "极点与极线 / Pole and Polar",
        "content": (
            "极点极线理论：给定圆和圆外一点 P（极点），\n"
            "从 P 向圆作两条切线，切点连线即为极线。\n"
            "PolePolar(P, l, O, r) 表示 P 关于圆 O 的极线为 l。\n"
            "规则：pole_polar_perp (OP⊥极线), pole_polar_tangent (极线上的点的切线过极点)。\n"
            "对偶原理：点在极线上 ⟺ 极点在该点的极线上。"
        ),
        "tags": ["projective", "pole_polar", "circle"],
    },
    {
        "title": "反演 / Inversion",
        "content": (
            "圆反演：以 O 为中心、r 为半径的反演将点 P 映射到 P'，\n"
            "使得 OP·OP'=r²。\n"
            "InvImage(P', P, O, r) 表示 P' 是 P 关于圆 O 的反演像。\n"
            "反演的主要性质：\n"
            "1. 过反演中心的直线映射为自身（inversion_collinear）\n"
            "2. 不过反演中心的圆映射为圆（inversion_circle_fixed）\n"
            "3. 反演保持角度（共形映射）\n"
            "4. 反演将相切关系映射为相切关系\n"
            "应用：化圆为线、Ptolemy 不等式、费尔巴赫定理。"
        ),
        "tags": ["inversion", "circle", "projective"],
    },
    {
        "title": "交比 / Cross Ratio",
        "content": (
            "交比 (A,B;C,D) = (AC·BD)/(BC·AD) 是射影不变量。\n"
            "EqCrossRatio(A,B,C,D,E,F,G,H) 表示两组四点交比相等。\n"
            "规则：cross_ratio_sym (交比对称性),\n"
            "cross_ratio_from_harmonic (调和点列 → 交比=-1)。\n"
            "射影变换保持交比不变。"
        ),
        "tags": ["cross_ratio", "projective", "harmonic"],
    },
    {
        "title": "根轴 / Radical Axis",
        "content": (
            "根轴定理：两圆的等幂点的轨迹是一条直线（根轴），\n"
            "且根轴垂直于两圆连心线。\n"
            "RadicalAxis(l, O1, r1, O2, r2) 表示 l 是两圆的根轴。\n"
            "规则：radical_axis_perp (根轴⊥连心线)。\n"
            "三圆的根轴交于一点（根心）。\n"
            "应用：证共圆、幂方程。"
        ),
        "tags": ["radical_axis", "circle", "power"],
    },
    # ── Concurrence & collinearity ───────────────────────────
    {
        "title": "三角形重心 / Centroid and Medians",
        "content": (
            "三角形的三条中线交于一点——重心。\n"
            "重心将每条中线分为 2:1。\n"
            "medians_concurrent 规则：给定三个 Midpoint，推出 Concurrent。\n"
            "重心坐标等于三个顶点坐标的算术平均。"
        ),
        "tags": ["triangle", "centroid", "concurrent"],
    },
    {
        "title": "塞瓦定理与梅涅劳斯定理 / Ceva's and Menelaus' Theorems",
        "content": (
            "塞瓦定理：若三角形 ABC 的三条塞瓦线 AD, BE, CF 共点，\n"
            "则 (AF/FB)(BD/DC)(CE/EA)=1。\n"
            "梅涅劳斯定理：若直线与三角形三边（或延长线）分别交于 D, E, F，\n"
            "则 (AF/FB)(BD/DC)(CE/EA)=-1。\n"
            "这两个定理是判断共线与共点的强有力工具。\n"
            "可结合 EqRatio、Collinear、Concurrent 谓词使用。"
        ),
        "tags": ["ceva", "menelaus", "concurrent", "collinear", "ratio"],
    },
    # ── Proof strategies ─────────────────────────────────────
    {
        "title": "几何证明策略：辅助线 / Proof Strategy: Auxiliary Lines",
        "content": (
            "常见辅助线策略：\n"
            "1. 作中点、中线、中位线 → 利用中位线定理\n"
            "2. 作垂线 → 利用勾股定理或垂直关系\n"
            "3. 作平行线 → 利用平行线性质（内错角、同位角）\n"
            "4. 作角平分线 → 利用角平分线定理\n"
            "5. 作外接圆 → 利用圆周角定理\n"
            "6. 作反演 → 将复杂圆关系化为直线关系\n"
            "选择策略时，看目标谓词：\n"
            "  Parallel → 找中位线或平行线传递\n"
            "  Cong → 找全等三角形或垂直平分线\n"
            "  EqAngle → 找圆周角或等腰三角形\n"
            "  Collinear → 找梅涅劳斯或调和点列"
        ),
        "tags": ["strategy", "auxiliary", "proof_technique"],
    },
    {
        "title": "几何证明策略：问题类型分析 / Problem Type Analysis",
        "content": (
            "根据前提和目标谓词选择策略：\n"
            "• Midpoint + Parallel → 中位线定理\n"
            "• Cyclic + EqAngle → 圆周角定理\n"
            "• Cong + Perpendicular → 垂直平分线定理\n"
            "• SimTri + Cong → 相似+等边⇒全等\n"
            "• Parallel + Perpendicular → 平行垂直传递\n"
            "• AngleBisect + Between → 角平分线定理(比例)\n"
            "• Multiple Midpoints → 中线共点(重心)\n"
            "• PolePolar + OnCircle → 极点极线切线关系\n"
            "• Harmonic + Collinear → 调和点列性质\n"
            "如果所有规则都不够：考虑辅助构造或跨域桥接。"
        ),
        "tags": ["strategy", "problem_type", "proof_technique"],
    },
    {
        "title": "等距与全等的关系 / EqDist and Cong",
        "content": (
            "EqDist(A, B, C) 表示 A 到 B 和 C 等距，即 AB=AC。\n"
            "Cong(A, B, C, D) 表示线段 AB 与线段 CD 全等。\n"
            "转换规则：eqdist_from_cong (Cong→EqDist), eqdist_to_cong (EqDist→Cong)。\n"
            "EqDist 常用于描述圆上的点（到圆心等距）或垂直平分线。"
        ),
        "tags": ["congruence", "equidistance"],
    },
    {
        "title": "面积相等 / Equal Area",
        "content": (
            "EqArea(A,B,C,D,E,F) 表示 △ABC 与 △DEF 面积相等。\n"
            "等积的来源：全等三角形 (congtri_eqarea), 等底等高。\n"
            "eqarea_sym 规则：面积相等的对称性。\n"
            "应用：面积法证明共线、比例关系。"
        ),
        "tags": ["area", "triangle", "congruent"],
    },
    {
        "title": "介于性与共线 / Betweenness and Collinearity",
        "content": (
            "Between(A, B, C) 表示 B 在线段 AC 之间。\n"
            "between_collinear 规则：介于性 ⇒ 三点共线。\n"
            "midpoint_between 规则：中点 ⇒ 介于性。\n"
            "应用：证明点在线段上、分割线段。"
        ),
        "tags": ["between", "collinear", "midpoint"],
    },
]


# ═══════════════════════════════════════════════════════════════════════
#  Module-level singleton
# ═══════════════════════════════════════════════════════════════════════

_global_rag: Optional[GeometryRAG] = None
_global_rag_lock = threading.Lock()


def get_rag(
    *,
    enable_web: bool = True,
    local_threshold: float = 0.4,
) -> GeometryRAG:
    """Return (or create) the process-wide singleton GeometryRAG.

    Parameters
    ----------
    enable_web : bool
        Whether to enable web search fallback.
    local_threshold : float
        Minimum local score to skip web search.

    Returns
    -------
    GeometryRAG
    """
    global _global_rag
    if _global_rag is None:
        with _global_rag_lock:
            if _global_rag is None:
                _global_rag = GeometryRAG(
                    enable_web=enable_web,
                    local_threshold=local_threshold,
                )
                logger.info("GeometryRAG initialised (web=%s)", enable_web)
    return _global_rag
