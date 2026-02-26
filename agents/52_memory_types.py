"""
Memory Types — Chapter 52
==========================
Four canonical memory systems for LLM agents, mirroring human cognition:

  1. Working Memory     — the active context window (sliding buffer of turns)
  2. Episodic Memory    — past conversation episodes, retrieved by recency/query
  3. Semantic Memory    — factual knowledge store, retrieved by embedding similarity
  4. Procedural Memory  — learned action/tool sequences for recurring tasks

Plus:
  5. UnifiedMemorySystem — combines all four, providing a single API

Cognitive mapping:
  ┌─────────────────────┬──────────────────────────────────────┐
  │ Memory Type         │ LLM Agent Analog                     │
  ├─────────────────────┼──────────────────────────────────────┤
  │ Working (STM)       │ Context window / KV-cache            │
  │ Episodic            │ Conversation logs / summaries        │
  │ Semantic            │ Vector DB / RAG knowledge base       │
  │ Procedural          │ Tool recipes / action templates      │
  └─────────────────────┴──────────────────────────────────────┘

Run standalone:
    python 52_memory_types.py
"""

from __future__ import annotations

import math
import time
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _approx_tokens(text: str) -> int:
    """~4 chars per token (rough GPT-4 estimate)."""
    return max(1, len(text) // 4)

def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def _mock_embed(text: str, dim: int = 32) -> List[float]:
    """
    Deterministic mock embedding: hash characters to a unit vector.
    Replace with a real embedding model (e.g. sentence-transformers) in production.
    """
    vec = [0.0] * dim
    for i, ch in enumerate(text):
        vec[ord(ch) % dim] += 1.0 / (i + 1)
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


# ──────────────────────────────────────────────────────────────────────────────
# 1. Working Memory
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Turn:
    role: str        # "user" | "assistant" | "tool"
    content: str
    timestamp: str = field(default_factory=_now_iso)
    tokens: int = 0

    def __post_init__(self):
        self.tokens = _approx_tokens(self.content)

    def to_message(self) -> dict:
        return {"role": self.role, "content": self.content}


class WorkingMemory:
    """
    Maintains the active portion of the context window.

    Strategies:
      - max_tokens: hard token budget — evict oldest turns when exceeded
      - max_turns:  hard turn budget  — evict oldest turns when exceeded
      - keep_system: always preserve the system prompt

    Think of this as the CPU's L1/L2 cache — fast, small, always current.
    """

    def __init__(
        self,
        max_tokens: int = 4096,
        max_turns: int = 20,
        system_prompt: str = "",
    ):
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.system_prompt = system_prompt
        self._turns: deque[Turn] = deque()
        self._system_tokens = _approx_tokens(system_prompt)

    # ── Mutation ─────────────────────────────────────────────────────────────

    def add(self, role: str, content: str):
        """Add a new turn and evict oldest if budget exceeded."""
        turn = Turn(role=role, content=content)
        self._turns.append(turn)
        self._evict()

    def _evict(self):
        """Remove oldest turns until within budget."""
        while self._turns and (
            self.token_count > self.max_tokens or len(self._turns) > self.max_turns
        ):
            self._turns.popleft()

    def clear(self):
        self._turns.clear()

    # ── Accessors ────────────────────────────────────────────────────────────

    @property
    def token_count(self) -> int:
        return self._system_tokens + sum(t.tokens for t in self._turns)

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    def to_messages(self) -> List[dict]:
        """Return full message list suitable for LLM API."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(t.to_message() for t in self._turns)
        return messages

    def last_n(self, n: int) -> List[Turn]:
        turns = list(self._turns)
        return turns[-n:]

    def utilization(self) -> float:
        """Return fraction of token budget used (0.0–1.0)."""
        return min(1.0, self.token_count / self.max_tokens)

    def __repr__(self):
        return (f"WorkingMemory(turns={self.turn_count}, "
                f"tokens={self.token_count}/{self.max_tokens}, "
                f"util={self.utilization():.0%})")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Episodic Memory
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Episode:
    """A saved conversation session or interaction chunk."""
    id: str
    summary: str            # Human-readable summary of the episode
    turns: List[dict]       # Full message list
    timestamp: str = field(default_factory=_now_iso)
    tags: List[str] = field(default_factory=list)
    importance: float = 0.5  # 0.0 = trivial, 1.0 = critical
    embedding: List[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.embedding:
            self.embedding = _mock_embed(self.summary)


class EpisodicMemory:
    """
    Stores and retrieves past conversation episodes.

    Retrieval strategies:
      - recency:    return the most recent N episodes
      - similarity: return episodes whose embeddings are closest to the query
      - importance: return highest-importance episodes
      - hybrid:     weighted combination of recency + similarity + importance

    Think of this as long-term autobiographical memory — "what happened before."
    """

    def __init__(self, max_episodes: int = 1000):
        self.max_episodes = max_episodes
        self._episodes: List[Episode] = []
        self._id_counter = 0

    # ── Storage ──────────────────────────────────────────────────────────────

    def save(
        self,
        summary: str,
        turns: List[dict],
        tags: List[str] = None,
        importance: float = 0.5,
    ) -> Episode:
        """Persist a new episode."""
        ep = Episode(
            id=f"ep_{self._id_counter:04d}",
            summary=summary,
            turns=turns,
            tags=tags or [],
            importance=importance,
        )
        self._id_counter += 1
        self._episodes.append(ep)
        if len(self._episodes) > self.max_episodes:
            # Evict lowest-importance episode
            self._episodes.sort(key=lambda e: e.importance)
            self._episodes.pop(0)
        return ep

    # ── Retrieval ────────────────────────────────────────────────────────────

    def retrieve_recent(self, n: int = 5) -> List[Episode]:
        return sorted(self._episodes, key=lambda e: e.timestamp, reverse=True)[:n]

    def retrieve_similar(self, query: str, top_k: int = 5) -> List[Tuple[Episode, float]]:
        query_emb = _mock_embed(query)
        scored = [
            (ep, _cosine_sim(query_emb, ep.embedding))
            for ep in self._episodes
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def retrieve_by_tag(self, tag: str) -> List[Episode]:
        return [ep for ep in self._episodes if tag in ep.tags]

    def retrieve_hybrid(self, query: str, top_k: int = 5) -> List[Episode]:
        """
        Hybrid retrieval: score = 0.4*similarity + 0.3*recency + 0.3*importance
        """
        query_emb = _mock_embed(query)
        now_ts = time.time()

        def _recency_score(ep: Episode) -> float:
            ep_ts = datetime.fromisoformat(ep.timestamp).timestamp()
            age_hours = max(0, (now_ts - ep_ts) / 3600)
            return 1.0 / (1.0 + age_hours)

        scored = []
        for ep in self._episodes:
            sim = _cosine_sim(query_emb, ep.embedding)
            rec = _recency_score(ep)
            combined = 0.4 * sim + 0.3 * rec + 0.3 * ep.importance
            scored.append((ep, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [ep for ep, _ in scored[:top_k]]

    def format_for_context(self, episodes: List[Episode]) -> str:
        """Format episodes into a text block for injection into context."""
        lines = ["=== Relevant Past Episodes ==="]
        for ep in episodes:
            lines.append(f"[{ep.id} | {ep.timestamp[:10]} | importance={ep.importance:.1f}]")
            lines.append(f"Summary: {ep.summary}")
            lines.append("")
        return "\n".join(lines)

    def __len__(self):
        return len(self._episodes)

    def __repr__(self):
        return f"EpisodicMemory(episodes={len(self._episodes)}, max={self.max_episodes})"


# ──────────────────────────────────────────────────────────────────────────────
# 3. Semantic Memory
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Fact:
    """A discrete piece of factual knowledge."""
    id: str
    content: str                # The actual fact text
    source: str = "unknown"     # Where this came from
    confidence: float = 1.0     # Model's confidence in this fact
    created_at: str = field(default_factory=_now_iso)
    embedding: List[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.embedding:
            self.embedding = _mock_embed(self.content)


class SemanticMemory:
    """
    Long-term factual knowledge store, retrieved by semantic similarity.

    This is the agent's "world knowledge" store — distinct from conversation
    history (episodic) or current context (working).

    Supports:
      - add / update / delete facts
      - similarity search over embeddings
      - confidence-weighted retrieval
      - deduplication via similarity threshold

    Think of this as the vector database / RAG knowledge base.
    """

    def __init__(self, dedup_threshold: float = 0.92):
        self._facts: Dict[str, Fact] = {}
        self._id_counter = 0
        self.dedup_threshold = dedup_threshold

    # ── Storage ──────────────────────────────────────────────────────────────

    def add(self, content: str, source: str = "unknown", confidence: float = 1.0) -> Optional[Fact]:
        """
        Add a new fact. Returns None if a near-duplicate already exists.
        """
        emb = _mock_embed(content)

        # Deduplication check
        for existing in self._facts.values():
            if _cosine_sim(emb, existing.embedding) >= self.dedup_threshold:
                return None  # Skip duplicate

        fact = Fact(
            id=f"fact_{self._id_counter:04d}",
            content=content,
            source=source,
            confidence=confidence,
            embedding=emb,
        )
        self._id_counter += 1
        self._facts[fact.id] = fact
        return fact

    def update(self, fact_id: str, content: str, confidence: float = None):
        if fact_id not in self._facts:
            raise KeyError(f"Fact '{fact_id}' not found.")
        fact = self._facts[fact_id]
        fact.content = content
        fact.embedding = _mock_embed(content)
        if confidence is not None:
            fact.confidence = confidence

    def delete(self, fact_id: str):
        self._facts.pop(fact_id, None)

    def add_many(self, facts: List[str], source: str = "batch"):
        return [self.add(f, source=source) for f in facts]

    # ── Retrieval ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5, min_confidence: float = 0.0) -> List[Tuple[Fact, float]]:
        """Return top-k facts most similar to query, above min_confidence."""
        query_emb = _mock_embed(query)
        scored = [
            (f, _cosine_sim(query_emb, f.embedding))
            for f in self._facts.values()
            if f.confidence >= min_confidence
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def format_for_context(self, facts: List[Tuple[Fact, float]]) -> str:
        lines = ["=== Retrieved Knowledge ==="]
        for fact, score in facts:
            lines.append(f"[{fact.id} | conf={fact.confidence:.2f} | sim={score:.3f}]")
            lines.append(f"  {fact.content}")
        return "\n".join(lines)

    def __len__(self):
        return len(self._facts)

    def __repr__(self):
        return f"SemanticMemory(facts={len(self._facts)}, dedup_thresh={self.dedup_threshold})"


# ──────────────────────────────────────────────────────────────────────────────
# 4. Procedural Memory
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Procedure:
    """
    A learned action sequence for a recurring task.

    Example:
        name: "web_research_summary"
        trigger: "summarize latest news about X"
        steps: ["web_search(X, 5)", "extract_key_facts(results)", "summarize(facts)"]
    """
    id: str
    name: str
    description: str
    trigger_patterns: List[str]     # Phrases that activate this procedure
    steps: List[str]                # Ordered action sequence
    success_count: int = 0
    failure_count: int = 0
    avg_reward: float = 0.0
    embedding: List[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.embedding:
            self.embedding = _mock_embed(self.description)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


class ProceduralMemory:
    """
    Stores learned action sequences (skills/recipes) for recurring tasks.

    The agent can:
      1. Look up whether a known procedure applies to the current task
      2. Record new procedures discovered during ReAct/planning
      3. Update success/failure counts to reinforce or deprecate procedures

    Think of this as muscle memory — automatic skill execution without
    deliberate step-by-step planning.
    """

    def __init__(self):
        self._procedures: Dict[str, Procedure] = {}
        self._id_counter = 0

    # ── Registration ─────────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        description: str,
        steps: List[str],
        trigger_patterns: List[str] = None,
    ) -> Procedure:
        proc = Procedure(
            id=f"proc_{self._id_counter:04d}",
            name=name,
            description=description,
            trigger_patterns=trigger_patterns or [],
            steps=steps,
        )
        self._id_counter += 1
        self._procedures[proc.id] = proc
        return proc

    def record_outcome(self, proc_id: str, success: bool, reward: float = 0.0):
        """Update success/failure stats after a procedure runs."""
        p = self._procedures[proc_id]
        if success:
            p.success_count += 1
        else:
            p.failure_count += 1
        # Running average reward
        total = p.success_count + p.failure_count
        p.avg_reward = p.avg_reward + (reward - p.avg_reward) / total

    # ── Retrieval ────────────────────────────────────────────────────────────

    def find_applicable(self, task: str, top_k: int = 3) -> List[Tuple[Procedure, float]]:
        """Find procedures that might apply to the given task."""
        task_emb = _mock_embed(task)
        scored = []
        for proc in self._procedures.values():
            sim = _cosine_sim(task_emb, proc.embedding)
            # Weight by success rate
            combined = 0.7 * sim + 0.3 * proc.success_rate
            scored.append((proc, combined))

        # Also check trigger patterns (keyword match)
        for proc, score in scored:
            for pattern in proc.trigger_patterns:
                if pattern.lower() in task.lower():
                    scored[scored.index((proc, score))] = (proc, min(1.0, score + 0.2))
                    break

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def format_procedure(self, proc: Procedure) -> str:
        lines = [
            f"Procedure: {proc.name} [{proc.id}]",
            f"Description: {proc.description}",
            f"Steps:",
        ]
        for i, step in enumerate(proc.steps, 1):
            lines.append(f"  {i}. {step}")
        lines.append(f"Success rate: {proc.success_rate:.0%} ({proc.success_count}/{proc.success_count+proc.failure_count})")
        return "\n".join(lines)

    def __len__(self):
        return len(self._procedures)

    def __repr__(self):
        return f"ProceduralMemory(procedures={len(self._procedures)})"


# ──────────────────────────────────────────────────────────────────────────────
# 5. Unified Memory System
# ──────────────────────────────────────────────────────────────────────────────

class UnifiedMemorySystem:
    """
    Unified API combining all four memory types.

    Usage pattern in an agent loop:
        1. memory.observe(user_turn)         — add to working memory
        2. context = memory.compose(task)    — retrieve relevant memories
        3. <call LLM with context>
        4. memory.observe(assistant_turn)    — add response
        5. memory.consolidate()             — save episode, extract facts

    This mirrors the human memory consolidation cycle:
        perception → working memory → long-term memory
    """

    def __init__(
        self,
        system_prompt: str = "",
        working_max_tokens: int = 4096,
        working_max_turns: int = 20,
    ):
        self.working   = WorkingMemory(working_max_tokens, working_max_turns, system_prompt)
        self.episodic  = EpisodicMemory()
        self.semantic  = SemanticMemory()
        self.procedural = ProceduralMemory()

        self._session_turns: List[dict] = []

    # ── Observe ──────────────────────────────────────────────────────────────

    def observe(self, role: str, content: str):
        """Add a new turn to working memory and session log."""
        self.working.add(role, content)
        self._session_turns.append({"role": role, "content": content})

    # ── Compose ──────────────────────────────────────────────────────────────

    def compose(self, task: str, inject_memory: bool = True) -> List[dict]:
        """
        Build the final message list for the LLM:
          [system] memories-injected
          [user/assistant turns from working memory]

        Retrieves relevant episodic + semantic + procedural memories and
        injects them into the system message.
        """
        memory_context_parts = []

        if inject_memory and len(self.episodic) > 0:
            episodes = self.episodic.retrieve_hybrid(task, top_k=3)
            if episodes:
                memory_context_parts.append(self.episodic.format_for_context(episodes))

        if inject_memory and len(self.semantic) > 0:
            facts = self.semantic.retrieve(task, top_k=5)
            if facts:
                memory_context_parts.append(self.semantic.format_for_context(facts))

        if inject_memory and len(self.procedural) > 0:
            procs = self.procedural.find_applicable(task, top_k=2)
            if procs:
                lines = ["=== Applicable Procedures ==="]
                for proc, score in procs:
                    lines.append(self.procedural.format_procedure(proc))
                    lines.append("")
                memory_context_parts.append("\n".join(lines))

        messages = self.working.to_messages()

        # Inject memories into the system message
        if memory_context_parts and messages and messages[0]["role"] == "system":
            injected = messages[0]["content"] + "\n\n" + "\n\n".join(memory_context_parts)
            messages[0] = {"role": "system", "content": injected}

        return messages

    # ── Consolidate ──────────────────────────────────────────────────────────

    def consolidate(self, summary: str, tags: List[str] = None, importance: float = 0.5):
        """
        End-of-session memory consolidation:
          - Save the current session as an episode
          - Extract facts from the session (simplified: just save the summary)
          - Clear the session buffer
        """
        if self._session_turns:
            self.episodic.save(
                summary=summary,
                turns=self._session_turns.copy(),
                tags=tags or [],
                importance=importance,
            )
            # Extract key facts (simplified)
            self.semantic.add(summary, source="consolidation")
            self._session_turns.clear()

    # ── Stats ────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "working":    str(self.working),
            "episodic":   str(self.episodic),
            "semantic":   str(self.semantic),
            "procedural": str(self.procedural),
            "session_turns": len(self._session_turns),
        }

    def __repr__(self):
        return (f"UnifiedMemorySystem(\n"
                f"  working={self.working}\n"
                f"  episodic={self.episodic}\n"
                f"  semantic={self.semantic}\n"
                f"  procedural={self.procedural}\n"
                f")")


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────

def demo():
    print("=" * 60)
    print("CHAPTER 52 — Memory Types Demo")
    print("=" * 60)

    # ── Working Memory ────────────────────────────────────────────────────────
    print("\n[1] Working Memory")
    wm = WorkingMemory(max_tokens=500, max_turns=10, system_prompt="You are a helpful AI.")
    for i in range(5):
        wm.add("user", f"User turn {i}: What is the meaning of life?")
        wm.add("assistant", f"Assistant turn {i}: The answer is 42, context {i}.")
    print(f"  {wm}")
    print(f"  Last 2 turns: {[t.role for t in wm.last_n(2)]}")
    print(f"  Context utilization: {wm.utilization():.0%}")

    # ── Episodic Memory ────────────────────────────────────────────────────────
    print("\n[2] Episodic Memory")
    em = EpisodicMemory()
    em.save("User asked about Python basics", [{"role": "user", "content": "What is Python?"}],
            tags=["python", "basics"], importance=0.4)
    em.save("Long discussion about RLHF training techniques", [], tags=["rlhf", "training"], importance=0.9)
    em.save("User needed help debugging a transformer model", [], tags=["debug", "transformer"], importance=0.7)

    print(f"  Episodes stored: {len(em)}")
    similar = em.retrieve_similar("transformer training", top_k=2)
    for ep, score in similar:
        print(f"  → [{ep.id}] sim={score:.3f}: {ep.summary[:50]}")

    # ── Semantic Memory ────────────────────────────────────────────────────────
    print("\n[3] Semantic Memory")
    sm = SemanticMemory()
    facts = [
        "Transformers use self-attention to model long-range dependencies.",
        "RLHF stands for Reinforcement Learning from Human Feedback.",
        "GPT-4 was released by OpenAI in March 2023.",
        "The Attention Is All You Need paper was published in 2017.",
        "Vector databases store embeddings for semantic search.",
    ]
    sm.add_many(facts, source="knowledge_base")
    sm.add("Near-duplicate of transformers fact", source="test")  # Should not be added (dedup)

    print(f"  Facts stored: {len(sm)}")
    results = sm.retrieve("how does attention work", top_k=2)
    for fact, score in results:
        print(f"  → [{fact.id}] sim={score:.3f}: {fact.content[:60]}")

    # ── Procedural Memory ──────────────────────────────────────────────────────
    print("\n[4] Procedural Memory")
    pm = ProceduralMemory()
    pm.register(
        name="web_research_pipeline",
        description="Research a topic using web search and summarize findings",
        steps=["web_search(query, 5)", "filter_relevant_results(results)", "summarize(results)"],
        trigger_patterns=["research", "find information", "look up"],
    )
    pm.register(
        name="code_debug_pipeline",
        description="Debug Python code by analyzing error messages and tracing",
        steps=["parse_error(traceback)", "locate_source(error)", "propose_fix(context)", "verify_fix(code)"],
        trigger_patterns=["debug", "error", "fix", "traceback"],
    )

    # Simulate some successes
    procs = list(pm._procedures.values())
    pm.record_outcome(procs[0].id, success=True, reward=0.9)
    pm.record_outcome(procs[0].id, success=True, reward=0.85)

    applicable = pm.find_applicable("I need to research the latest LLM papers", top_k=2)
    for proc, score in applicable:
        print(f"  → [{proc.id}] score={score:.3f}: {proc.name} (success={proc.success_rate:.0%})")

    # ── Unified System ─────────────────────────────────────────────────────────
    print("\n[5] Unified Memory System")
    ums = UnifiedMemorySystem(
        system_prompt="You are a helpful AI assistant.",
        working_max_tokens=2000,
        working_max_turns=10,
    )

    # Seed with some knowledge
    ums.semantic.add("Python is a high-level programming language.", source="seed")
    ums.procedural.register("quick_answer", "Answer directly without tools", ["what is", "explain"])

    # Simulate a conversation
    ums.observe("user", "What is Python?")
    ums.observe("assistant", "Python is a high-level, interpreted programming language.")
    ums.observe("user", "How do I install packages?")
    ums.observe("assistant", "Use pip: pip install <package-name>")

    messages = ums.compose(task="What is Python?", inject_memory=True)
    print(f"  Messages in composed context: {len(messages)}")

    ums.consolidate(
        summary="User asked about Python basics and package management.",
        tags=["python", "beginner"],
        importance=0.5,
    )
    print(f"  After consolidation: {ums.stats()}")


if __name__ == "__main__":
    demo()
