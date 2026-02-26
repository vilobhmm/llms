"""
Agent Orchestrator — Chapter 55
=================================
Wires together all agent components into a production-ready agent:

  ┌─────────────────────────────────────────────────────────────────────┐
  │                         Agent Loop                                  │
  │                                                                     │
  │  User Input                                                         │
  │     │                                                               │
  │     ▼                                                               │
  │  Memory.compose()   ←─── EpisodicMemory + SemanticMemory           │
  │     │                    + ProceduralMemory injected                │
  │     ▼                                                               │
  │  Select Pattern     ←─── CoT / ReAct / Plan-and-Execute /          │
  │     │                    Reflection / ToT / Multi-Agent            │
  │     ▼                                                               │
  │  Model Call ──────────────────────────────────────────────────────► │
  │     │                                                               │
  │     ▼                                                               │
  │  Tool Dispatch      ←─── ToolRegistry + ToolDispatcher             │
  │     │                                                               │
  │     ▼                                                               │
  │  Working Memory.add(response)                                       │
  │     │                                                               │
  │     ▼                                                               │
  │  Context Debugger   ←─── Snapshot + Visualizer (debug mode)        │
  │     │                                                               │
  │     ▼                                                               │
  │  Memory.consolidate()                                               │
  │     │                                                               │
  │     ▼                                                               │
  │  Response to User                                                   │
  └─────────────────────────────────────────────────────────────────────┘

Run standalone:
    python 55_agent_orchestrator.py
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# ── Internal imports ─────────────────────────────────────────────────────────
# (All from same agents/ directory)
try:
    from agents._loader import load_agent_module
    _tool_use     = load_agent_module("50_tool_use")
    _patterns     = load_agent_module("51_agentic_patterns")
    _memory_types = load_agent_module("52_memory_types")
    _mem_stores   = load_agent_module("53_memory_store")
    _debugger     = load_agent_module("54_context_debugger")
except Exception:
    # Direct import fallback when running in the agents/ directory
    import importlib.util, os, sys

    def _load(filename: str):
        path = os.path.join(os.path.dirname(__file__), filename)
        spec = importlib.util.spec_from_file_location(filename, path)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[filename] = mod   # register before exec so dataclasses work
        spec.loader.exec_module(mod)
        return mod

    _tool_use     = _load("50_tool_use.py")
    _patterns     = _load("51_agentic_patterns.py")
    _memory_types = _load("52_memory_types.py")
    _mem_stores   = _load("53_memory_store.py")
    _debugger     = _load("54_context_debugger.py")

# Pull commonly-used names into local scope
ToolRegistry     = _tool_use.ToolRegistry
ToolDispatcher   = _tool_use.ToolDispatcher
ToolCall         = _tool_use.ToolCall
TOOL_REGISTRY    = _tool_use.TOOL_REGISTRY

UnifiedMemorySystem = _memory_types.UnifiedMemorySystem
WorkingMemory       = _memory_types.WorkingMemory

ContextAnalyzer  = _debugger.ContextAnalyzer
ContextVisualizer = _debugger.ContextVisualizer
MemoryStateInspector = _debugger.MemoryStateInspector

CoT    = _patterns.ChainOfThought
ReAct  = _patterns.ReActAgent
PlanEx = _patterns.PlanAndExecute
Reflect = _patterns.ReflectionAgent
mock_model = _patterns.mock_model


# ──────────────────────────────────────────────────────────────────────────────
# Agent configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    """
    Full configuration for the orchestrated agent.

    model_fn:         Your LLM callable (messages → str).
    pattern:          Which agentic loop to use ("react"|"cot"|"plan"|"reflect"|"direct").
    system_prompt:    Base system prompt (before memory injection).
    max_context_tokens: Context window size.
    working_max_tokens: Token budget for working memory.
    working_max_turns:  Turn budget for working memory.
    debug:            If True, print context snapshots and memory state each turn.
    consolidate_every: Consolidate memory every N turns (0 = never).
    tools:            Tool schemas to pass to the model (in addition to registry).
    """
    model_fn: Callable
    pattern: str = "react"          # react | cot | plan | reflect | direct
    system_prompt: str = "You are a helpful AI agent."
    max_context_tokens: int = 128_000
    working_max_tokens: int = 8_192
    working_max_turns: int = 40
    debug: bool = False
    consolidate_every: int = 10
    tools: List[dict] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Turn result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TurnResult:
    """The outcome of a single agent turn."""
    turn_id: int
    user_input: str
    response: str
    tool_calls_made: List[str]       # Names of tools called
    pattern_used: str
    elapsed_ms: float
    memory_tokens: int
    context_utilization: float


# ──────────────────────────────────────────────────────────────────────────────
# Main Agent
# ──────────────────────────────────────────────────────────────────────────────

class Agent:
    """
    The fully orchestrated agent.

    Lifecycle:
        agent = Agent(config)
        result = agent.chat("What is the capital of France?")
        result = agent.chat("Now search for its population.")
        agent.debug_context()     # Print context window state
        agent.debug_memory()      # Print memory system state
    """

    def __init__(self, config: AgentConfig, registry: ToolRegistry = None):
        self.config   = config
        self.registry = registry or TOOL_REGISTRY
        self.dispatcher = ToolDispatcher(self.registry)

        self.memory = UnifiedMemorySystem(
            system_prompt=config.system_prompt,
            working_max_tokens=config.working_max_tokens,
            working_max_turns=config.working_max_turns,
        )

        # Seed procedural memory with known pipelines
        self.memory.procedural.register(
            name="web_research",
            description="Research a topic using web search",
            steps=["web_search(topic, 5)", "summarize_results(results)"],
            trigger_patterns=["search", "find", "research", "look up"],
        )
        self.memory.procedural.register(
            name="calculation",
            description="Perform a mathematical calculation",
            steps=["calculator(expression)", "format_result(number)"],
            trigger_patterns=["calculate", "compute", "what is", "how much"],
        )

        # Pattern instances (lazy-built per turn to inject current model_fn)
        self._analyzer = ContextAnalyzer(config.max_context_tokens)
        self._viz      = ContextVisualizer(use_color=True)
        self._inspector = MemoryStateInspector()

        self._turn_count = 0
        self._history: List[TurnResult] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def chat(self, user_input: str) -> str:
        """
        Main entry point for a single conversation turn.
        Returns the agent's final text response.
        """
        t0 = time.perf_counter()
        self._turn_count += 1
        turn_id = self._turn_count

        # 1. Add user input to working memory
        self.memory.observe("user", user_input)

        # 2. Retrieve relevant memories and compose context
        composed = self.memory.compose(task=user_input, inject_memory=True)

        # 3. (Optional) debug: print context snapshot
        if self.config.debug:
            self._print_context_snapshot(composed, turn_id)

        # 4. Run the selected agentic pattern
        pattern = self.config.pattern
        response, tool_names = self._run_pattern(pattern, user_input, composed)

        # 5. Add response to working memory
        self.memory.observe("assistant", response)

        # 6. (Optional) consolidate memory every N turns
        if (self.config.consolidate_every > 0
                and self._turn_count % self.config.consolidate_every == 0):
            self._consolidate(user_input, response)

        elapsed = (time.perf_counter() - t0) * 1000
        snap = self._analyzer.analyze(self.memory.working.to_messages(), label=f"turn_{turn_id}")

        result = TurnResult(
            turn_id=turn_id,
            user_input=user_input,
            response=response,
            tool_calls_made=tool_names,
            pattern_used=pattern,
            elapsed_ms=elapsed,
            memory_tokens=snap.total_tokens,
            context_utilization=snap.utilization,
        )
        self._history.append(result)
        return response

    # ── Pattern dispatch ──────────────────────────────────────────────────────

    def _run_pattern(
        self,
        pattern: str,
        user_input: str,
        composed: List[dict],
    ) -> tuple[str, List[str]]:
        """Run the selected pattern and return (response, tool_names_called)."""
        model_fn = self.config.model_fn
        tool_names: List[str] = []

        if pattern == "direct":
            response = model_fn(composed)

        elif pattern == "cot":
            cot = CoT(model_fn)
            result = cot.run(user_input)
            response = result["answer"]

        elif pattern == "react":
            # Build tool dict for ReAct
            react_tools = {
                name: (lambda n=name: lambda args: self._dispatch_tool(n, args))()
                for name in self.registry.list_tools()
            }
            react = ReAct(model_fn, react_tools, max_steps=6)
            result = react.run(user_input)
            response = result["answer"] or "No answer produced."
            tool_names = [
                s.action for s in result["steps"]
                if s.action and not s.is_final
            ]

        elif pattern == "plan":
            pe = PlanEx(model_fn)
            result = pe.run(user_input)
            response = result["final_result"] or "Plan completed."

        elif pattern == "reflect":
            ra = Reflect(model_fn, num_rounds=2)
            result = ra.run(user_input)
            response = result["final"]

        else:
            response = model_fn(composed)

        return response, tool_names

    def _dispatch_tool(self, tool_name: str, args: dict) -> str:
        """Execute a tool and return the string result."""
        call = ToolCall(id=str(uuid.uuid4())[:8], name=tool_name, arguments=args)
        result = self.dispatcher.execute(call)
        return str(result.content) if not result.is_error else f"[Error] {result.error}"

    # ── Memory consolidation ──────────────────────────────────────────────────

    def _consolidate(self, user_input: str, response: str):
        """Compress and archive the current session into long-term memory."""
        summary = (
            f"Turn {self._turn_count}: User asked about '{user_input[:60]}...'. "
            f"Agent responded: '{response[:80]}...'"
        )
        self.memory.consolidate(
            summary=summary,
            tags=self._infer_tags(user_input),
            importance=self._infer_importance(user_input, response),
        )

    def _infer_tags(self, text: str) -> List[str]:
        tag_keywords = {
            "python": ["python", "code", "script"],
            "math":   ["calculate", "compute", "formula", "equation"],
            "search": ["search", "find", "look up", "research"],
            "llm":    ["llm", "model", "transformer", "gpt", "claude"],
        }
        tags = []
        lower = text.lower()
        for tag, kws in tag_keywords.items():
            if any(kw in lower for kw in kws):
                tags.append(tag)
        return tags or ["general"]

    def _infer_importance(self, user_input: str, response: str) -> float:
        """Assign higher importance to longer, more detailed exchanges."""
        length_score = min(1.0, (len(user_input) + len(response)) / 2000)
        return 0.3 + 0.4 * length_score

    # ── Debug utilities ───────────────────────────────────────────────────────

    def debug_context(self, label: str = None) -> str:
        """Print and return the current context window analysis."""
        messages = self.memory.working.to_messages()
        snap = self._analyzer.analyze(
            messages,
            tools=self.config.tools,
            label=label or f"turn_{self._turn_count}",
        )
        report = self._viz.full_report(snap)
        print(report)
        return report

    def debug_memory(self) -> str:
        """Print and return full memory system state."""
        out = ["\n" + "=" * 60, "  MEMORY STATE INSPECTOR", "=" * 60]
        out.append(self._inspector.inspect(self.memory.working,    "working_memory"))
        out.append(self._inspector.inspect(self.memory.episodic,   "episodic_memory"))
        out.append(self._inspector.inspect(self.memory.semantic,   "semantic_memory"))
        out.append(self._inspector.inspect(self.memory.procedural, "procedural_memory"))
        result = "\n".join(out)
        print(result)
        return result

    def debug_tools(self) -> str:
        """Print all registered tools and their schemas."""
        lines = ["\n" + "=" * 60, "  REGISTERED TOOLS", "=" * 60]
        for schema in self.registry.all_schemas():
            params = list(schema["parameters"]["properties"].keys())
            lines.append(f"  • {schema['name']}({', '.join(params)})")
            lines.append(f"    {schema['description']}")
        result = "\n".join(lines)
        print(result)
        return result

    def _print_context_snapshot(self, messages: List[dict], turn_id: int):
        snap = self._analyzer.analyze(messages, tools=self.config.tools, label=f"turn_{turn_id}")
        print(self._viz.full_report(snap))

    # ── Session summary ───────────────────────────────────────────────────────

    def session_summary(self) -> str:
        if not self._history:
            return "No turns recorded."
        lines = [
            "=" * 60,
            "  SESSION SUMMARY",
            "=" * 60,
            f"  Total turns: {len(self._history)}",
            f"  Pattern: {self.config.pattern}",
            "",
            f"  {'Turn':<6} {'Tokens':>8} {'Util':>6} {'ms':>8} {'Tools'}",
            "  " + "-" * 50,
        ]
        for r in self._history:
            tools = ", ".join(r.tool_calls_made[:3]) or "none"
            lines.append(
                f"  {r.turn_id:<6} {r.memory_tokens:>8,} {r.context_utilization:>5.1%} "
                f"{r.elapsed_ms:>8.0f} {tools}"
            )
        total_tokens = sum(r.memory_tokens for r in self._history)
        lines.append(f"\n  Total memory tokens accumulated: {total_tokens:,}")
        return "\n".join(lines)

    def __repr__(self):
        return (f"Agent(pattern={self.config.pattern!r}, "
                f"turns={self._turn_count}, "
                f"working={self.memory.working})")


# ──────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ──────────────────────────────────────────────────────────────────────────────

def build_agent(
    model_fn: Callable = None,
    pattern: str = "react",
    system_prompt: str = "You are a helpful AI agent.",
    debug: bool = False,
    **kwargs,
) -> Agent:
    """
    Quick factory for building an Agent with sensible defaults.

    Args:
        model_fn: LLM callable. Defaults to mock_model for testing.
        pattern:  "direct" | "cot" | "react" | "plan" | "reflect"
        system_prompt: System instructions.
        debug:    Enable context/memory debug output each turn.

    Example:
        agent = build_agent(model_fn=my_claude_fn, pattern="react")
        print(agent.chat("Search for the latest AI research."))
    """
    config = AgentConfig(
        model_fn=model_fn or mock_model,
        pattern=pattern,
        system_prompt=system_prompt,
        debug=debug,
        **kwargs,
    )
    return Agent(config)


# ──────────────────────────────────────────────────────────────────────────────
# Anthropic Claude integration (optional — requires anthropic package)
# ──────────────────────────────────────────────────────────────────────────────

def make_claude_fn(
    model: str = "claude-sonnet-4-6",
    api_key: str = None,
    max_tokens: int = 2048,
) -> Callable:
    """
    Create a model_fn that calls the Anthropic Claude API.

    Usage:
        claude_fn = make_claude_fn(api_key="sk-ant-...")
        agent = build_agent(model_fn=claude_fn, pattern="react")
        print(agent.chat("What is the capital of Japan?"))
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")

    client = anthropic.Anthropic(api_key=api_key)

    def _call(messages: List[dict]) -> str:
        # Separate system from user/assistant messages
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system  = [m for m in messages if m.get("role") != "system"]
        system_text = system_msgs[0]["content"] if system_msgs else ""

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_text,
            messages=non_system,
        )
        return response.content[0].text

    return _call


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────

def demo():
    print("=" * 70)
    print("CHAPTER 55 — Agent Orchestrator Demo")
    print("=" * 70)

    # ── Build agent ───────────────────────────────────────────────────────────
    agent = build_agent(
        model_fn=mock_model,
        pattern="react",
        system_prompt="You are a knowledgeable AI assistant with memory and tools.",
        debug=False,   # Set True to print context snapshot each turn
    )

    # ── Seed semantic memory ──────────────────────────────────────────────────
    agent.memory.semantic.add(
        "Transformers use multi-head self-attention to model long-range dependencies.",
        source="knowledge_base",
        confidence=0.98,
    )
    agent.memory.semantic.add(
        "RLHF is used to align language models with human preferences.",
        source="knowledge_base",
        confidence=0.97,
    )

    # ── Conversation ──────────────────────────────────────────────────────────
    conversations = [
        "What is the transformer architecture?",
        "Calculate sqrt(196) + 2**8",
        "Search for recent papers on memory in LLMs.",
        "Summarize what we've discussed so far.",
    ]

    print("\n── Conversation ──────────────────────────────────────────")
    for user_msg in conversations:
        print(f"\nUser: {user_msg}")
        response = agent.chat(user_msg)
        print(f"Agent: {response[:150]}...")

    # ── Debug context ─────────────────────────────────────────────────────────
    print("\n── Context Debug ─────────────────────────────────────────")
    agent.debug_context()

    # ── Debug memory ──────────────────────────────────────────────────────────
    print("\n── Memory Debug ──────────────────────────────────────────")
    agent.debug_memory()

    # ── Debug tools ───────────────────────────────────────────────────────────
    print("\n── Tools ─────────────────────────────────────────────────")
    agent.debug_tools()

    # ── Session summary ───────────────────────────────────────────────────────
    print("\n── Session Summary ───────────────────────────────────────")
    print(agent.session_summary())

    # ── Pattern comparison ────────────────────────────────────────────────────
    print("\n── Pattern Comparison ────────────────────────────────────")
    patterns = ["direct", "cot", "react", "plan", "reflect"]
    task = "Explain what makes transformer models effective."
    for pat in patterns:
        a = build_agent(model_fn=mock_model, pattern=pat)
        t0 = time.perf_counter()
        resp = a.chat(task)
        ms = (time.perf_counter() - t0) * 1000
        print(f"  [{pat:<8}] {ms:6.0f}ms → {resp[:60]}...")

    print("\nOrchestrator demo complete.")
    print("\nTo use with Claude API:")
    print("  claude_fn = make_claude_fn(api_key='your-key')")
    print("  agent = build_agent(claude_fn, pattern='react', debug=True)")
    print("  agent.chat('Your question here')")


if __name__ == "__main__":
    demo()
