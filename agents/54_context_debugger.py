"""
Context Debugger — Chapter 54
==============================
Inspect, visualize, and diff the full LLM context at any point in time.

What it shows:
  ┌─────────────────────────────────────────────────────┐
  │  SECTION          │ TOKENS │ % BUDGET │ CONTENT     │
  ├───────────────────┼────────┼──────────┼─────────────┤
  │  system_prompt    │   128  │   3.1%   │ "You are…"  │
  │  tool_definitions │   512  │  12.5%   │ [calc, …]   │
  │  injected_memory  │   256  │   6.3%   │ "Past eps…" │
  │  history          │  2048  │  50.0%   │ 12 turns    │
  │  current_user     │    64  │   1.6%   │ "What is…"  │
  └───────────────────┴────────┴──────────┴─────────────┘

Features:
  1. ContextSnapshot    — capture context state as a typed object
  2. TokenCounter       — count tokens per section (tiktoken or approx)
  3. ContextAnalyzer    — compute budgets, roles, content stats
  4. ContextDiff        — diff two snapshots to see what changed
  5. ContextVisualizer  — ASCII bar chart + text report
  6. MemoryStateInspector — inspect UnifiedMemorySystem contents

Run standalone:
    python 54_context_debugger.py
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Token counting
# ──────────────────────────────────────────────────────────────────────────────

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")   # GPT-4 encoding
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:           # type: ignore[misc]
        """Approximate token count when tiktoken is unavailable (~4 chars/token)."""
        return max(1, len(text) // 4)


def count_message_tokens(messages: List[dict]) -> int:
    """Count tokens for a full message list, including role overhead (~4 tokens/msg)."""
    total = 0
    for msg in messages:
        total += 4  # Per-message overhead (role + separators)
        total += count_tokens(str(msg.get("content", "")))
        # Tool definitions nested in content
        if isinstance(msg.get("content"), list):
            for part in msg["content"]:
                total += count_tokens(str(part))
    total += 2  # Reply priming
    return total


# ──────────────────────────────────────────────────────────────────────────────
# 1. ContextSnapshot
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ContextSection:
    name: str
    content: str                   # String representation of section content
    tokens: int
    raw: Any = None                # Original object (dict, list, etc.)


@dataclass
class ContextSnapshot:
    """
    A frozen snapshot of the full LLM context at a single point in time.

    Sections:
      - system_prompt:    The [system] message content
      - tool_definitions: The list of tool schemas passed to the model
      - injected_memory:  Memory content injected into system or first user msg
      - history:          All [user]/[assistant]/[tool] turns before current
      - current_user:     The latest [user] message being sent now
    """
    label: str                         # Snapshot name (e.g., "turn_3")
    sections: Dict[str, ContextSection]
    max_context_tokens: int = 128_000
    timestamp: str = ""

    @property
    def total_tokens(self) -> int:
        return sum(s.tokens for s in self.sections.values())

    @property
    def utilization(self) -> float:
        return self.total_tokens / max(1, self.max_context_tokens)

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_context_tokens - self.total_tokens)

    def section(self, name: str) -> Optional[ContextSection]:
        return self.sections.get(name)

    def __repr__(self):
        return (f"ContextSnapshot(label={self.label!r}, "
                f"tokens={self.total_tokens:,}, "
                f"util={self.utilization:.1%})")


# ──────────────────────────────────────────────────────────────────────────────
# 2. ContextAnalyzer — build a ContextSnapshot from a message list
# ──────────────────────────────────────────────────────────────────────────────

MEMORY_MARKERS = [
    "=== Relevant Past Episodes ===",
    "=== Retrieved Knowledge ===",
    "=== Applicable Procedures ===",
    "=== Historical Overview ===",
    "=== Recent Summaries ===",
]


class ContextAnalyzer:
    """
    Parse a raw message list + optional tool schemas into a ContextSnapshot.

    It separates the message list into logical sections:
      - system_prompt (and any injected memory within it)
      - tool_definitions
      - history
      - current_user
    """

    def __init__(self, max_context_tokens: int = 128_000):
        self.max_context_tokens = max_context_tokens

    def _split_system(self, system_content: str) -> Tuple[str, str]:
        """Split a system message into (base_prompt, injected_memory)."""
        for marker in MEMORY_MARKERS:
            if marker in system_content:
                parts = system_content.split(marker, 1)
                base = parts[0].rstrip()
                memory_part = marker + parts[1]
                return base, memory_part
        return system_content, ""

    def analyze(
        self,
        messages: List[dict],
        tools: List[dict] = None,
        label: str = "snapshot",
    ) -> ContextSnapshot:
        tools = tools or []
        sections: Dict[str, ContextSection] = {}

        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system  = [m for m in messages if m.get("role") != "system"]

        # ── System prompt + injected memory ───────────────────────────────
        if system_msgs:
            raw_sys = system_msgs[0].get("content", "")
            base_sys, mem_content = self._split_system(raw_sys)
        else:
            base_sys, mem_content = "", ""

        sections["system_prompt"] = ContextSection(
            name="system_prompt",
            content=base_sys,
            tokens=count_tokens(base_sys),
            raw=system_msgs,
        )

        if mem_content:
            sections["injected_memory"] = ContextSection(
                name="injected_memory",
                content=mem_content,
                tokens=count_tokens(mem_content),
                raw=mem_content,
            )

        # ── Tool definitions ───────────────────────────────────────────────
        if tools:
            tools_str = json.dumps(tools, indent=2)
            sections["tool_definitions"] = ContextSection(
                name="tool_definitions",
                content=tools_str,
                tokens=count_tokens(tools_str),
                raw=tools,
            )

        # ── History (all turns except the last user message) ───────────────
        if non_system:
            history = non_system[:-1] if non_system[-1].get("role") == "user" else non_system
            current = non_system[-1] if non_system[-1].get("role") == "user" else None
        else:
            history, current = [], None

        if history:
            history_str = "\n".join(
                f"[{m['role']}]: {str(m.get('content',''))[:200]}" for m in history
            )
            sections["history"] = ContextSection(
                name="history",
                content=history_str,
                tokens=count_message_tokens(history),
                raw=history,
            )

        if current:
            cur_content = str(current.get("content", ""))
            sections["current_user"] = ContextSection(
                name="current_user",
                content=cur_content,
                tokens=count_tokens(cur_content),
                raw=current,
            )

        return ContextSnapshot(
            label=label,
            sections=sections,
            max_context_tokens=self.max_context_tokens,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 3. ContextDiff — compare two snapshots
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SectionDiff:
    section: str
    tokens_before: int
    tokens_after: int
    delta: int
    added: bool       # Section newly added
    removed: bool     # Section removed


@dataclass
class ContextDiff:
    label_before: str
    label_after: str
    section_diffs: List[SectionDiff]
    total_delta: int

    def summary(self) -> str:
        lines = [
            f"Diff: {self.label_before!r} → {self.label_after!r}",
            f"Total token delta: {self.total_delta:+,}",
        ]
        for d in self.section_diffs:
            status = "[NEW]" if d.added else "[DEL]" if d.removed else ""
            lines.append(f"  {d.section:<20} {d.tokens_before:>6} → {d.tokens_after:>6}  ({d.delta:+,}) {status}")
        return "\n".join(lines)


def diff_snapshots(before: ContextSnapshot, after: ContextSnapshot) -> ContextDiff:
    """Compute the diff between two context snapshots."""
    all_sections = set(before.sections) | set(after.sections)
    diffs = []

    for name in sorted(all_sections):
        tb = before.sections[name].tokens if name in before.sections else 0
        ta = after.sections[name].tokens  if name in after.sections  else 0
        diffs.append(SectionDiff(
            section=name,
            tokens_before=tb,
            tokens_after=ta,
            delta=ta - tb,
            added=name not in before.sections,
            removed=name not in after.sections,
        ))

    return ContextDiff(
        label_before=before.label,
        label_after=after.label,
        section_diffs=diffs,
        total_delta=after.total_tokens - before.total_tokens,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 4. ContextVisualizer — render ASCII reports
# ──────────────────────────────────────────────────────────────────────────────

# ANSI colours for terminal output
_COLORS = {
    "system_prompt":    "\033[94m",  # Blue
    "tool_definitions": "\033[92m",  # Green
    "injected_memory":  "\033[95m",  # Magenta
    "history":          "\033[93m",  # Yellow
    "current_user":     "\033[96m",  # Cyan
    "reset":            "\033[0m",
    "bold":             "\033[1m",
}

BAR_WIDTH = 40


class ContextVisualizer:
    """
    Render a ContextSnapshot as:
      1. A table (section / tokens / % budget / content preview)
      2. An ASCII horizontal stacked-bar chart
      3. A section content dump
    """

    def __init__(self, use_color: bool = True, width: int = 80):
        self.use_color = use_color
        self.width = width

    def _c(self, section: str) -> str:
        if not self.use_color:
            return ""
        return _COLORS.get(section, "") or ""

    def _reset(self) -> str:
        return _COLORS["reset"] if self.use_color else ""

    def _bold(self, text: str) -> str:
        if self.use_color:
            return _COLORS["bold"] + text + _COLORS["reset"]
        return text

    # ── Table ────────────────────────────────────────────────────────────────

    def table(self, snap: ContextSnapshot) -> str:
        lines = []
        title = f" Context Snapshot: {snap.label} "
        lines.append(self._bold("=" * self.width))
        lines.append(self._bold(title.center(self.width)))
        lines.append(self._bold("=" * self.width))
        lines.append(
            f"{'Section':<22} {'Tokens':>8} {'%Budget':>8} {'Preview'}"
        )
        lines.append("-" * self.width)

        for name, sec in snap.sections.items():
            pct = sec.tokens / max(1, snap.max_context_tokens) * 100
            preview = sec.content.replace("\n", " ")[:40]
            color = self._c(name)
            reset = self._reset()
            lines.append(f"{color}{name:<22}{reset} {sec.tokens:>8,} {pct:>7.1f}%   {preview}")

        lines.append("-" * self.width)
        total_pct = snap.utilization * 100
        remaining = snap.remaining_tokens
        lines.append(
            f"{'TOTAL':<22} {snap.total_tokens:>8,} {total_pct:>7.1f}%"
        )
        lines.append(
            f"{'REMAINING':<22} {remaining:>8,} {100 - total_pct:>7.1f}%"
        )
        lines.append("=" * self.width)
        return "\n".join(lines)

    # ── Bar chart ────────────────────────────────────────────────────────────

    def bar_chart(self, snap: ContextSnapshot) -> str:
        """Render a horizontal stacked bar representing token budget usage."""
        lines = [self._bold("Context Window Usage")]
        lines.append(f"[0{'─' * (BAR_WIDTH - 2)}{snap.max_context_tokens:,}]")

        bar = []
        symbols = {
            "system_prompt":    "S",
            "tool_definitions": "T",
            "injected_memory":  "M",
            "history":          "H",
            "current_user":     "U",
        }
        legend = []

        for name, sec in snap.sections.items():
            sym = symbols.get(name, "?")
            width = max(1, round(sec.tokens / snap.max_context_tokens * BAR_WIDTH))
            color = self._c(name)
            reset = self._reset()
            bar.append(color + sym * width + reset)
            legend.append(f"  {color}{sym}{reset} {name} ({sec.tokens:,} tokens)")

        remaining = snap.remaining_tokens
        rem_width = max(0, BAR_WIDTH - sum(
            max(1, round(s.tokens / snap.max_context_tokens * BAR_WIDTH))
            for s in snap.sections.values()
        ))
        bar.append("." * rem_width)

        lines.append("[" + "".join(bar) + "]")
        lines.extend(legend)
        lines.append(f"  . remaining ({remaining:,} tokens, {snap.utilization:.1%} used)")
        return "\n".join(lines)

    # ── Content dump ─────────────────────────────────────────────────────────

    def dump_section(self, snap: ContextSnapshot, section_name: str, max_chars: int = 500) -> str:
        sec = snap.section(section_name)
        if not sec:
            return f"[Section '{section_name}' not found in snapshot '{snap.label}']"

        lines = [
            self._bold(f"━━━ {section_name} ({sec.tokens:,} tokens) ━━━"),
            sec.content[:max_chars],
        ]
        if len(sec.content) > max_chars:
            lines.append(f"... [{len(sec.content) - max_chars} more chars]")
        return "\n".join(lines)

    def full_report(self, snap: ContextSnapshot) -> str:
        parts = [
            self.table(snap),
            "",
            self.bar_chart(snap),
        ]
        return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# 5. MemoryStateInspector
# ──────────────────────────────────────────────────────────────────────────────

class MemoryStateInspector:
    """
    Inspect the full state of any memory object from ch52 or ch53.

    Supports:
      - WorkingMemory
      - EpisodicMemory / EpisodicStore
      - SemanticMemory / InMemoryVectorStore
      - ProceduralMemory
      - KeyValueStore
      - SlidingWindowBuffer
      - UnifiedMemorySystem
    """

    def inspect(self, obj: Any, label: str = None) -> str:
        name = label or type(obj).__name__
        method = f"_inspect_{type(obj).__name__}"
        if hasattr(self, method):
            return getattr(self, method)(obj, name)
        return self._generic(obj, name)

    def _section(self, title: str, lines: List[str]) -> str:
        out = [f"\n{'━'*50}", f"  {title}", f"{'━'*50}"]
        out.extend(lines)
        return "\n".join(out)

    # ── Working Memory ───────────────────────────────────────────────────────

    def _inspect_WorkingMemory(self, obj: Any, name: str) -> str:
        lines = [
            f"  Turns:      {obj.turn_count}",
            f"  Tokens:     {obj.token_count:,} / {obj.max_tokens:,}",
            f"  Utilization:{obj.utilization():.1%}",
            f"  System:     {obj.system_prompt[:60]!r}...",
        ]
        for i, turn in enumerate(obj.last_n(3)):
            lines.append(f"  Turn[-{3-i}] [{turn.role}]: {turn.content[:60]!r}...")
        return self._section(f"WorkingMemory ({name})", lines)

    # ── Episodic Memory ──────────────────────────────────────────────────────

    def _inspect_EpisodicMemory(self, obj: Any, name: str) -> str:
        lines = [f"  Total episodes: {len(obj)}"]
        for ep in sorted(obj._episodes, key=lambda e: e.importance, reverse=True)[:3]:
            lines.append(f"  [{ep.id}] imp={ep.importance:.1f} tags={ep.tags}: {ep.summary[:50]!r}")
        return self._section(f"EpisodicMemory ({name})", lines)

    def _inspect_EpisodicStore(self, obj: Any, name: str) -> str:
        lines = [f"  Total episodes: {len(obj)}"]
        for ep in obj.by_importance(3):
            lines.append(f"  [{ep.id}] imp={ep.importance:.1f} age={ep.age_hours():.1f}h: {ep.summary[:50]!r}")
        return self._section(f"EpisodicStore ({name})", lines)

    # ── Semantic Memory ──────────────────────────────────────────────────────

    def _inspect_SemanticMemory(self, obj: Any, name: str) -> str:
        lines = [f"  Total facts: {len(obj)}"]
        for fact in list(obj._facts.values())[:5]:
            lines.append(f"  [{fact.id}] conf={fact.confidence:.2f}: {fact.content[:60]!r}")
        return self._section(f"SemanticMemory ({name})", lines)

    def _inspect_InMemoryVectorStore(self, obj: Any, name: str) -> str:
        lines = [f"  Total docs: {len(obj)}"]
        for entry in list(obj._entries.values())[:5]:
            cats = entry.metadata.get("cat", "?")
            lines.append(f"  [{entry.id}] cat={cats}: {entry.content[:60]!r}")
        return self._section(f"InMemoryVectorStore ({name})", lines)

    # ── Procedural Memory ────────────────────────────────────────────────────

    def _inspect_ProceduralMemory(self, obj: Any, name: str) -> str:
        lines = [f"  Total procedures: {len(obj)}"]
        for proc in obj._procedures.values():
            lines.append(f"  [{proc.id}] {proc.name}: {proc.success_rate:.0%} success, {len(proc.steps)} steps")
        return self._section(f"ProceduralMemory ({name})", lines)

    # ── Key-Value Store ──────────────────────────────────────────────────────

    def _inspect_KeyValueStore(self, obj: Any, name: str) -> str:
        obj.expire_all()
        lines = [f"  Total entries: {len(obj)} / {obj.max_size}"]
        for key in list(obj._store.keys())[:10]:
            entry = obj._store[key]
            ttl_info = f"TTL={entry.ttl}s" if entry.ttl else "no TTL"
            val_preview = str(entry.value)[:50]
            lines.append(f"  {key!r}: {val_preview!r} ({ttl_info})")
        return self._section(f"KeyValueStore ({name})", lines)

    # ── Sliding Window Buffer ────────────────────────────────────────────────

    def _inspect_SlidingWindowBuffer(self, obj: Any, name: str) -> str:
        lines = [
            f"  Items:      {len(obj)} / {obj.max_items}",
            f"  Tokens:     {obj._total_tokens:,} / {obj.max_tokens:,}",
            f"  Utilization:{obj.utilization():.1%}",
            f"  Policy:     {obj.policy}",
        ]
        for entry in obj.peek(3):
            preview = str(entry.content)[:50]
            lines.append(f"  → {preview!r}")
        return self._section(f"SlidingWindowBuffer ({name})", lines)

    # ── Unified Memory System ────────────────────────────────────────────────

    def _inspect_UnifiedMemorySystem(self, obj: Any, name: str) -> str:
        lines = [
            "  Components:",
            "  " + str(obj.working),
            "  " + str(obj.episodic),
            "  " + str(obj.semantic),
            "  " + str(obj.procedural),
            f"  Session turns: {len(obj._session_turns)}",
        ]
        return self._section(f"UnifiedMemorySystem ({name})", lines)

    def _generic(self, obj: Any, name: str) -> str:
        lines = [f"  Type: {type(obj).__name__}", f"  Repr: {repr(obj)[:200]}"]
        return self._section(name, lines)


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────

def _build_sample_messages() -> Tuple[List[dict], List[dict]]:
    """Build sample messages for demo."""
    tools = [
        {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
        {
            "name": "calculator",
            "description": "Evaluate a math expression",
            "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
        },
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant with access to tools.\n\n"
                "=== Relevant Past Episodes ===\n"
                "[ep_0001 | 2025-02-20 | importance=0.8]\n"
                "Summary: User asked about Python transformers library.\n\n"
                "=== Retrieved Knowledge ===\n"
                "[fact_0001 | conf=0.95 | sim=0.87]\n"
                "  Transformers use self-attention mechanisms.\n"
            ),
        },
        {"role": "user",      "content": "What is the transformer architecture?"},
        {"role": "assistant", "content": "The transformer architecture is based on self-attention mechanisms..."},
        {"role": "user",      "content": "Can you search for recent papers on it?"},
        {"role": "assistant", "content": "I'll search for recent transformer papers.", "tool_calls": [{"id": "tc1", "function": {"name": "web_search", "arguments": '{"query": "transformer architecture 2025"}'}}]},
        {"role": "tool",      "content": "[Result 1] Transformers in 2025...", "tool_call_id": "tc1", "name": "web_search"},
        {"role": "assistant", "content": "Here are the recent papers I found..."},
        {"role": "user",      "content": "What is 2^10 + sqrt(256)?"},  # This is the "current" user message
    ]
    return messages, tools


def demo():
    print("=" * 80)
    print("CHAPTER 54 — Context Debugger Demo")
    print("=" * 80)

    messages, tools = _build_sample_messages()

    # ── Build snapshot ────────────────────────────────────────────────────────
    analyzer = ContextAnalyzer(max_context_tokens=8192)
    snap = analyzer.analyze(messages, tools=tools, label="turn_4")
    print(f"\nSnapshot: {snap}")

    # ── Full report ───────────────────────────────────────────────────────────
    viz = ContextVisualizer(use_color=True, width=80)
    print("\n" + viz.full_report(snap))

    # ── Section dump ──────────────────────────────────────────────────────────
    print("\n" + viz.dump_section(snap, "system_prompt", max_chars=200))
    print("\n" + viz.dump_section(snap, "injected_memory", max_chars=200))
    print("\n" + viz.dump_section(snap, "current_user"))

    # ── Diff two snapshots ────────────────────────────────────────────────────
    messages2 = messages + [
        {"role": "assistant", "content": "2^10 + sqrt(256) = 1024 + 16 = 1040"},
        {"role": "user",      "content": "Great! Can you now explain RLHF?"},
    ]
    snap2 = analyzer.analyze(messages2, tools=tools, label="turn_5")
    diff = diff_snapshots(snap, snap2)
    print("\n" + diff.summary())

    # ── Memory State Inspector (using real classes from ch52/53) ─────────────
    print("\n" + "=" * 80)
    print("Memory State Inspector")
    print("=" * 80)

    import importlib.util, sys as _sys, os as _os
    def _load_ch(fname):
        path = _os.path.join(_os.path.dirname(__file__), fname)
        spec = importlib.util.spec_from_file_location(fname, path)
        mod  = importlib.util.module_from_spec(spec)
        _sys.modules[fname] = mod   # register BEFORE exec so dataclasses work
        spec.loader.exec_module(mod)
        return mod

    ch52 = _load_ch("52_memory_types.py")
    ch53 = _load_ch("53_memory_store.py")

    # Build real memory objects
    wm = ch52.WorkingMemory(max_tokens=4096, max_turns=20, system_prompt="You are a helpful assistant.")
    for role, text in [("user", "What is Python?"), ("assistant", "Python is a programming language."),
                       ("user", "How does attention work?"), ("assistant", "Attention computes weighted sums.")]:
        wm.add(role, text)

    em = ch52.EpisodicMemory()
    em.save("User asked about Python basics", [], tags=["python"], importance=0.6)
    em.save("Deep dive into transformer attention", [], tags=["transformer", "attention"], importance=0.9)

    sm = ch52.SemanticMemory()
    sm.add("Python is a high-level interpreted language.", source="kb", confidence=0.99)
    sm.add("Attention mechanisms compute weighted averages over value vectors.", source="kb", confidence=0.98)

    pm = ch52.ProceduralMemory()
    pm.register("search_and_summarize", "Web search then summarize results",
                ["web_search(q)", "summarize(results)"], trigger_patterns=["research", "find"])
    pm.record_outcome(list(pm._procedures.keys())[0], success=True, reward=0.9)
    pm.record_outcome(list(pm._procedures.keys())[0], success=True, reward=0.85)

    kv = ch53.KeyValueStore(max_size=100, default_ttl=None)
    kv.set("user_name", "Alice")
    kv.set("session_id", "sess_abc123")
    kv.set("last_tool", "calculator")

    buf = ch53.SlidingWindowBuffer(max_tokens=500, max_items=10)
    for i in range(6):
        buf.push(f"Turn {i}: This is a conversation turn about topic {i}.")

    inspector = MemoryStateInspector()
    print(inspector.inspect(wm,  "WorkingMemory"))
    print(inspector.inspect(em,  "EpisodicMemory"))
    print(inspector.inspect(sm,  "SemanticMemory"))
    print(inspector.inspect(pm,  "ProceduralMemory"))
    print(inspector.inspect(kv,  "KeyValueStore"))
    print(inspector.inspect(buf, "SlidingWindowBuffer"))

    print("\nContext debugger demo complete.")


if __name__ == "__main__":
    demo()
