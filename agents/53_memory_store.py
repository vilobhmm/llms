"""
Memory Stores — Chapter 53
===========================
Low-level storage backends that power the memory types in Chapter 52.

  1. InMemoryVectorStore     — cosine-similarity vector retrieval (mini RAG)
  2. SlidingWindowBuffer     — fixed-size FIFO buffer for context management
  3. EpisodicStore           — time-indexed episode log with multi-field search
  4. KeyValueStore           — fast exact-key lookup with TTL expiry
  5. HierarchicalSummaryStore— compress old context into summaries (memory compression)

These are intentionally simple, dependency-free implementations.
In production: swap for Faiss, Chroma, Pinecone, Redis, etc.

Run standalone:
    python 53_memory_store.py
"""

from __future__ import annotations

import math
import time
import json
import heapq
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Tuple, TypeVar

V = TypeVar("V")

# ──────────────────────────────────────────────────────────────────────────────
# Embedding utilities (same mock as ch52, factored out here)
# ──────────────────────────────────────────────────────────────────────────────

def mock_embed(text: str, dim: int = 64) -> List[float]:
    """Deterministic mock embedding — replace with real model in production."""
    vec = [0.0] * dim
    for i, ch in enumerate(text):
        vec[ord(ch) % dim] += 1.0 / (i + 1)
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]

def cosine_sim(a: List[float], b: List[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    na    = math.sqrt(sum(x * x for x in a))
    nb    = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 1. InMemoryVectorStore
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VectorEntry:
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class InMemoryVectorStore:
    """
    A simple vector store backed by a Python list.

    Supports:
      - upsert: add or update documents by ID
      - search: retrieve top-K nearest neighbours by cosine similarity
      - delete: remove a document
      - filter_search: search within a metadata-filtered subset

    Complexity: O(n) for search (brute-force). Use Faiss/Annoy for large corpora.

    Usage:
        store = InMemoryVectorStore(embed_fn=my_model.embed)
        store.upsert("doc1", "The sky is blue.", metadata={"category": "facts"})
        results = store.search("What color is the sky?", top_k=3)
    """

    def __init__(self, embed_fn: Callable[[str], List[float]] = None, dim: int = 64):
        self._embed = embed_fn or (lambda t: mock_embed(t, dim))
        self._entries: Dict[str, VectorEntry] = {}

    def upsert(self, doc_id: str, content: str, metadata: dict = None) -> VectorEntry:
        """Insert or update a document."""
        emb = self._embed(content)
        entry = VectorEntry(id=doc_id, content=content, embedding=emb, metadata=metadata or {})
        self._entries[doc_id] = entry
        return entry

    def upsert_many(self, docs: List[Tuple[str, str]], metadata: List[dict] = None):
        """Batch upsert."""
        metadata = metadata or [{}] * len(docs)
        return [self.upsert(doc_id, content, meta)
                for (doc_id, content), meta in zip(docs, metadata)]

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        metadata_filter: Dict[str, Any] = None,
    ) -> List[Tuple[VectorEntry, float]]:
        """
        Search for top-K documents most similar to the query.

        Args:
            query: Natural-language query string
            top_k: Maximum results to return
            min_score: Minimum cosine similarity threshold
            metadata_filter: Only return docs whose metadata contains these k/v pairs
        """
        query_emb = self._embed(query)
        candidates = self._entries.values()

        # Apply metadata filter
        if metadata_filter:
            candidates = [
                e for e in candidates
                if all(e.metadata.get(k) == v for k, v in metadata_filter.items())
            ]

        scored = [
            (entry, cosine_sim(query_emb, entry.embedding))
            for entry in candidates
        ]
        scored = [(e, s) for e, s in scored if s >= min_score]

        # Use heapq for efficient top-K selection
        top = heapq.nlargest(top_k, scored, key=lambda x: x[1])
        return top

    def delete(self, doc_id: str) -> bool:
        return self._entries.pop(doc_id, None) is not None

    def get(self, doc_id: str) -> Optional[VectorEntry]:
        return self._entries.get(doc_id)

    def __len__(self):
        return len(self._entries)

    def __repr__(self):
        return f"InMemoryVectorStore(docs={len(self._entries)})"


# ──────────────────────────────────────────────────────────────────────────────
# 2. SlidingWindowBuffer
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BufferEntry:
    content: Any
    tokens: int
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SlidingWindowBuffer(Generic[V]):
    """
    Fixed-capacity FIFO buffer with token-budget eviction.

    Two eviction policies:
      - "fifo":     always evict the oldest item
      - "priority": evict lowest-priority item (metadata["priority"])

    Use cases:
      - Sliding window context (context compression)
      - Rate-limiting tool call history
      - Caching recent observations

    Args:
        max_tokens:   Hard token ceiling (evict when exceeded)
        max_items:    Hard item count ceiling
        policy:       "fifo" or "priority"
    """

    def __init__(
        self,
        max_tokens: int = 4096,
        max_items: int = 50,
        policy: str = "fifo",
    ):
        self.max_tokens = max_tokens
        self.max_items = max_items
        self.policy = policy
        self._buffer: List[BufferEntry] = []
        self._total_tokens = 0

    def push(self, content: V, tokens: int = None, metadata: dict = None) -> BufferEntry:
        """Add an item to the buffer, evicting as needed."""
        tokens = tokens or self._estimate_tokens(content)
        entry = BufferEntry(content=content, tokens=tokens, metadata=metadata or {})
        self._buffer.append(entry)
        self._total_tokens += tokens
        self._evict()
        return entry

    def _estimate_tokens(self, content: Any) -> int:
        if isinstance(content, str):
            return max(1, len(content) // 4)
        return 1

    def _evict(self):
        while self._buffer and (
            self._total_tokens > self.max_tokens or len(self._buffer) > self.max_items
        ):
            if self.policy == "priority":
                # Evict lowest-priority item
                idx = min(range(len(self._buffer)),
                          key=lambda i: self._buffer[i].metadata.get("priority", 0))
            else:  # fifo
                idx = 0
            removed = self._buffer.pop(idx)
            self._total_tokens -= removed.tokens

    def peek(self, n: int = None) -> List[BufferEntry]:
        """Return last n items (or all if n is None)."""
        return self._buffer[-n:] if n else self._buffer[:]

    def clear(self):
        self._buffer.clear()
        self._total_tokens = 0

    def utilization(self) -> float:
        return min(1.0, self._total_tokens / self.max_tokens)

    def __len__(self):
        return len(self._buffer)

    def __iter__(self) -> Iterator[BufferEntry]:
        return iter(self._buffer)

    def __repr__(self):
        return (f"SlidingWindowBuffer(items={len(self._buffer)}, "
                f"tokens={self._total_tokens}/{self.max_tokens}, "
                f"util={self.utilization():.0%})")


# ──────────────────────────────────────────────────────────────────────────────
# 3. EpisodicStore
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StoredEpisode:
    id: str
    summary: str
    payload: Any                    # Full conversation, JSON, etc.
    tags: List[str]
    importance: float
    timestamp: float = field(default_factory=time.time)
    embedding: List[float] = field(default_factory=list)

    def age_hours(self) -> float:
        return (time.time() - self.timestamp) / 3600

    def __lt__(self, other: "StoredEpisode"):
        return self.importance < other.importance


class EpisodicStore:
    """
    Persistent episodic memory store with multi-dimensional retrieval.

    Retrieval dimensions:
      - Recency: most recently created
      - Importance: highest importance score
      - Semantic similarity: embedding distance to query
      - Tag match: exact tag filtering
      - Hybrid: weighted combination

    Capacity management:
      - max_episodes: hard cap; evicts by lowest importance when full
    """

    def __init__(self, max_episodes: int = 500, embed_fn: Callable = None):
        self._episodes: Dict[str, StoredEpisode] = OrderedDict()
        self._id_counter = 0
        self.max_episodes = max_episodes
        self._embed = embed_fn or mock_embed

    def save(
        self,
        summary: str,
        payload: Any = None,
        tags: List[str] = None,
        importance: float = 0.5,
    ) -> StoredEpisode:
        ep = StoredEpisode(
            id=f"ep_{self._id_counter:05d}",
            summary=summary,
            payload=payload,
            tags=tags or [],
            importance=importance,
            embedding=self._embed(summary),
        )
        self._id_counter += 1
        self._episodes[ep.id] = ep
        self._prune()
        return ep

    def _prune(self):
        while len(self._episodes) > self.max_episodes:
            # Evict lowest-importance episode
            evict_id = min(self._episodes, key=lambda k: self._episodes[k].importance)
            del self._episodes[evict_id]

    # ── Retrieval ────────────────────────────────────────────────────────────

    def by_recency(self, n: int = 5) -> List[StoredEpisode]:
        return sorted(self._episodes.values(), key=lambda e: e.timestamp, reverse=True)[:n]

    def by_importance(self, n: int = 5) -> List[StoredEpisode]:
        return sorted(self._episodes.values(), key=lambda e: e.importance, reverse=True)[:n]

    def by_tag(self, tag: str) -> List[StoredEpisode]:
        return [e for e in self._episodes.values() if tag in e.tags]

    def by_similarity(self, query: str, top_k: int = 5) -> List[Tuple[StoredEpisode, float]]:
        q_emb = self._embed(query)
        scored = [(e, cosine_sim(q_emb, e.embedding)) for e in self._episodes.values()]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

    def hybrid(
        self,
        query: str,
        top_k: int = 5,
        weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    ) -> List[Tuple[StoredEpisode, float]]:
        """
        Hybrid retrieval: score = w_sim * similarity + w_imp * importance + w_rec * recency_score
        weights: (similarity, importance, recency)
        """
        w_sim, w_imp, w_rec = weights
        q_emb = self._embed(query)
        now = time.time()

        scored = []
        for ep in self._episodes.values():
            sim = cosine_sim(q_emb, ep.embedding)
            imp = ep.importance
            rec = 1.0 / (1.0 + ep.age_hours())  # More recent = higher score
            combined = w_sim * sim + w_imp * imp + w_rec * rec
            scored.append((ep, combined))

        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

    def get(self, ep_id: str) -> Optional[StoredEpisode]:
        return self._episodes.get(ep_id)

    def delete(self, ep_id: str) -> bool:
        return bool(self._episodes.pop(ep_id, None))

    def __len__(self):
        return len(self._episodes)

    def __repr__(self):
        return f"EpisodicStore(episodes={len(self._episodes)}, max={self.max_episodes})"


# ──────────────────────────────────────────────────────────────────────────────
# 4. KeyValueStore (with TTL)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class KVEntry:
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    ttl: Optional[float] = None   # seconds; None = never expires

    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl


class KeyValueStore:
    """
    Simple key-value memory store with optional TTL expiration.

    Use cases:
      - Caching tool call results (e.g., API responses)
      - Storing agent session variables (user name, preferences, etc.)
      - Short-term scratch pad for intermediate reasoning
      - Tool output caching to avoid redundant calls

    Args:
        max_size: Maximum number of entries (evicts LRU when full)
        default_ttl: Default expiry in seconds (None = no expiry)
    """

    def __init__(self, max_size: int = 1000, default_ttl: Optional[float] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._store: OrderedDict[str, KVEntry] = OrderedDict()

    def set(self, key: str, value: Any, ttl: Optional[float] = ...) -> KVEntry:
        """Set a key-value pair. ttl overrides default_ttl if provided."""
        effective_ttl = self.default_ttl if ttl is ... else ttl
        entry = KVEntry(key=key, value=value, ttl=effective_ttl)

        if key in self._store:
            del self._store[key]
        self._store[key] = entry

        # LRU eviction
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

        return entry

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value. Returns default if missing or expired."""
        if key not in self._store:
            return default
        entry = self._store[key]
        if entry.is_expired():
            del self._store[key]
            return default
        # Move to end (LRU order)
        self._store.move_to_end(key)
        return entry.value

    def delete(self, key: str) -> bool:
        return bool(self._store.pop(key, None))

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def get_or_set(self, key: str, factory: Callable[[], Any], ttl: float = None) -> Any:
        """
        Return cached value if present; otherwise call factory and cache the result.
        Classic cache-aside pattern.
        """
        value = self.get(key)
        if value is None:
            value = factory()
            self.set(key, value, ttl=ttl)
        return value

    def expire_all(self):
        """Remove all expired entries."""
        expired = [k for k, v in self._store.items() if v.is_expired()]
        for k in expired:
            del self._store[k]

    def keys(self) -> List[str]:
        self.expire_all()
        return list(self._store.keys())

    def __len__(self):
        self.expire_all()
        return len(self._store)

    def __repr__(self):
        return f"KeyValueStore(entries={len(self._store)}, max={self.max_size})"


# ──────────────────────────────────────────────────────────────────────────────
# 5. HierarchicalSummaryStore
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SummaryLevel:
    level: int          # 0 = raw turns, 1 = chunk summaries, 2 = meta-summaries
    content: str
    source_ids: List[str]
    token_count: int
    created_at: float = field(default_factory=time.time)


class HierarchicalSummaryStore:
    """
    Compresses conversation history using hierarchical summarization.

    Architecture:
      Level 0: Raw turns           (e.g., 1000 turns, 8000 tokens)
      Level 1: Chunk summaries     (e.g., chunks of 20 turns → 200-token summary each)
      Level 2: Meta-summaries      (e.g., summaries of summaries)

    This mirrors how humans compress memories:
      Short-term → semantic encoding → long-term gist

    The agent always reads from the highest level first; retrieves raw turns
    only when details are needed.

    Args:
        chunk_size: How many items per chunk before summarizing
        summarize_fn: Callable(items: list[str]) -> str — your LLM summarizer
    """

    def __init__(self, chunk_size: int = 20, summarize_fn: Callable = None):
        self.chunk_size = chunk_size
        self._summarize = summarize_fn or self._default_summarize
        self._levels: Dict[int, List[SummaryLevel]] = {0: [], 1: [], 2: []}
        self._pending: List[str] = []   # Unsummarized raw turns

    @staticmethod
    def _default_summarize(items: List[str]) -> str:
        """Placeholder summarizer — replace with real LLM call."""
        n = len(items)
        preview = items[0][:60] if items else ""
        return f"[Summary of {n} items] Starting with: {preview}..."

    def push(self, content: str):
        """Add a new raw item. Triggers summarization when chunk_size is reached."""
        self._pending.append(content)
        tokens = max(1, len(content) // 4)

        raw_entry = SummaryLevel(
            level=0,
            content=content,
            source_ids=[],
            token_count=tokens,
        )
        self._levels[0].append(raw_entry)

        if len(self._pending) >= self.chunk_size:
            self._compress_pending()

    def _compress_pending(self):
        """Summarize pending raw turns into a Level-1 chunk summary."""
        if not self._pending:
            return

        summary_text = self._summarize(self._pending)
        source_ids = [f"raw_{i}" for i in range(len(self._pending))]

        chunk = SummaryLevel(
            level=1,
            content=summary_text,
            source_ids=source_ids,
            token_count=max(1, len(summary_text) // 4),
        )
        self._levels[1].append(chunk)
        self._pending.clear()

        # Level 2: summarize every N level-1 summaries
        if len(self._levels[1]) >= self.chunk_size:
            self._compress_level1()

    def _compress_level1(self):
        """Summarize Level-1 summaries into a Level-2 meta-summary."""
        level1_texts = [s.content for s in self._levels[1]]
        meta_summary = self._summarize(level1_texts)
        meta = SummaryLevel(
            level=2,
            content=meta_summary,
            source_ids=[s.source_ids[0] for s in self._levels[1] if s.source_ids],
            token_count=max(1, len(meta_summary) // 4),
        )
        self._levels[2].append(meta)
        self._levels[1].clear()

    def get_compressed_context(self, recent_raw: int = 10) -> str:
        """
        Build a compressed context string for injection into the LLM prompt.
        Returns: meta-summaries + recent chunk summaries + last N raw turns
        """
        parts = []

        if self._levels[2]:
            parts.append("=== Historical Overview ===")
            for s in self._levels[2]:
                parts.append(s.content)

        if self._levels[1]:
            parts.append("\n=== Recent Summaries ===")
            for s in self._levels[1][-3:]:  # Last 3 chunk summaries
                parts.append(s.content)

        if self._pending:
            parts.append("\n=== Recent Turns ===")
            for turn in self._pending[-recent_raw:]:
                parts.append(turn)

        return "\n".join(parts)

    def token_savings(self) -> dict:
        raw_tokens = sum(s.token_count for s in self._levels[0])
        l1_tokens  = sum(s.token_count for s in self._levels[1])
        l2_tokens  = sum(s.token_count for s in self._levels[2])
        compressed = l1_tokens + l2_tokens + sum(max(1, len(t)//4) for t in self._pending)
        return {
            "raw_tokens": raw_tokens,
            "compressed_tokens": compressed,
            "compression_ratio": raw_tokens / max(1, compressed),
            "savings_pct": max(0, (1 - compressed / max(1, raw_tokens)) * 100),
        }

    def __repr__(self):
        stats = self.token_savings()
        return (f"HierarchicalSummaryStore("
                f"raw={len(self._levels[0])}, "
                f"l1={len(self._levels[1])}, "
                f"l2={len(self._levels[2])}, "
                f"compression={stats['compression_ratio']:.1f}x)")


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────

def demo():
    print("=" * 60)
    print("CHAPTER 53 — Memory Stores Demo")
    print("=" * 60)

    # ── 1. Vector Store ──────────────────────────────────────────────────────
    print("\n[1] InMemoryVectorStore")
    vs = InMemoryVectorStore()
    docs = [
        ("d1", "Python is a high-level programming language."),
        ("d2", "Transformers use self-attention mechanisms."),
        ("d3", "Neural networks learn from data."),
        ("d4", "LLMs are trained on internet-scale corpora."),
        ("d5", "Vector databases enable semantic search."),
    ]
    vs.upsert_many(docs, metadata=[{"cat": "lang"}, {"cat": "ml"}, {"cat": "ml"}, {"cat": "llm"}, {"cat": "db"}])
    results = vs.search("how do language models work", top_k=3)
    print(f"  Stored: {len(vs)} docs")
    for entry, score in results:
        print(f"  → [{entry.id}] sim={score:.3f}: {entry.content[:50]}")

    # With metadata filter
    filtered = vs.search("machine learning", top_k=3, metadata_filter={"cat": "ml"})
    print(f"  ML-filtered results: {len(filtered)}")

    # ── 2. Sliding Window Buffer ─────────────────────────────────────────────
    print("\n[2] SlidingWindowBuffer")
    buf = SlidingWindowBuffer(max_tokens=200, max_items=5)
    for i in range(10):
        buf.push(f"Turn {i}: This is a conversation turn with some content.", metadata={"priority": i})
    print(f"  {buf}")
    print(f"  Last 3 items: {[e.content[:30] for e in buf.peek(3)]}")

    # ── 3. Episodic Store ────────────────────────────────────────────────────
    print("\n[3] EpisodicStore")
    es = EpisodicStore()
    for i in range(8):
        es.save(
            f"Session {i}: Discussed {'transformers' if i%2==0 else 'memory architectures'}",
            payload={"turns": i * 3},
            tags=["transformers" if i % 2 == 0 else "memory"],
            importance=0.3 + i * 0.09,
        )
    print(f"  Total episodes: {len(es)}")

    print("  By recency (top 3):")
    for ep in es.by_recency(3):
        print(f"    [{ep.id}] {ep.summary[:50]}")

    print("  By similarity to 'transformer attention':")
    for ep, score in es.by_similarity("transformer attention", top_k=3):
        print(f"    [{ep.id}] sim={score:.3f}: {ep.summary[:50]}")

    # ── 4. Key-Value Store ───────────────────────────────────────────────────
    print("\n[4] KeyValueStore")
    kv = KeyValueStore(max_size=100, default_ttl=60)
    kv.set("user_name", "Alice")
    kv.set("session_id", "sess_abc123")
    kv.set("temp_result", {"value": 42}, ttl=1)  # expires in 1 second

    print(f"  user_name: {kv.get('user_name')}")
    print(f"  session_id: {kv.get('session_id')}")
    print(f"  temp_result (before expire): {kv.get('temp_result')}")
    time.sleep(1.1)
    print(f"  temp_result (after expire): {kv.get('temp_result', 'EXPIRED')}")

    # Cache-aside pattern
    def expensive_computation():
        return {"result": "computed"}
    cached = kv.get_or_set("computation", expensive_computation, ttl=30)
    print(f"  Cached computation: {cached}")
    print(f"  Total KV entries: {len(kv)}")

    # ── 5. Hierarchical Summary Store ────────────────────────────────────────
    print("\n[5] HierarchicalSummaryStore")
    hss = HierarchicalSummaryStore(chunk_size=5)

    # Push 25 turns (enough to trigger multi-level summarization)
    for i in range(25):
        hss.push(f"Turn {i}: User asked about concept {i}. AI responded with detailed explanation #{i}.")

    print(f"  {hss}")
    stats = hss.token_savings()
    print(f"  Token savings: {stats['savings_pct']:.0f}% ({stats['compression_ratio']:.1f}x compression)")
    compressed = hss.get_compressed_context(recent_raw=3)
    print(f"  Compressed context preview:\n{compressed[:300]}...")


if __name__ == "__main__":
    demo()
