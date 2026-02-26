"""
Tool Use — Chapter 50: Function Calling & Tool Dispatch
========================================================
Covers the full lifecycle of tool use in LLM agents:

  1. Tool definition via JSON Schema
  2. ToolRegistry  — register, lookup, validate tools
  3. ToolCall      — a pending function invocation from the model
  4. ToolResult    — the outcome returned to the model
  5. Dispatcher    — parallel / sequential tool execution
  6. Example tools — calculator, web_search, get_weather, read_file,
                     python_eval, memory_lookup

Run standalone:
    python 50_tool_use.py
"""

from __future__ import annotations

import json
import math
import time
import traceback
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# 1. JSON-Schema type helpers
# ──────────────────────────────────────────────────────────────────────────────

def _py_type_to_json(annotation) -> str:
    """Map Python type annotations → JSON Schema type strings."""
    mapping = {
        int: "integer", float: "number", str: "string",
        bool: "boolean", list: "array", dict: "object",
    }
    return mapping.get(annotation, "string")


def schema_from_function(fn: Callable) -> dict:
    """
    Auto-generate a JSON Schema tool definition from a Python function's
    signature and docstring. Works best with type-annotated functions.

    Example output:
    {
      "name": "add",
      "description": "Add two numbers together.",
      "parameters": {
        "type": "object",
        "properties": {
          "a": {"type": "number"},
          "b": {"type": "number"}
        },
        "required": ["a", "b"]
      }
    }
    """
    sig = inspect.signature(fn)
    hints = fn.__annotations__
    doc = inspect.getdoc(fn) or ""

    properties: Dict[str, dict] = {}
    required: List[str] = []

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        prop: Dict[str, Any] = {"type": _py_type_to_json(hints.get(name, str))}
        # Grab inline description from first matching line in docstring
        for line in doc.splitlines():
            if name in line and ":" in line:
                prop["description"] = line.split(":", 1)[-1].strip()
                break
        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "name": fn.__name__,
        "description": doc.splitlines()[0] if doc else fn.__name__,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. ToolRegistry
# ──────────────────────────────────────────────────────────────────────────────

class ToolRegistry:
    """
    Central registry for all tools available to the agent.

    Usage:
        registry = ToolRegistry()

        @registry.register
        def add(a: float, b: float) -> float:
            "Add two numbers."
            return a + b

        result = registry.call("add", {"a": 1, "b": 2})  # 3.0
    """

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, dict] = {}

    # ── Registration ─────────────────────────────────────────────────────────

    def register(self, fn: Callable = None, *, schema: dict = None):
        """Decorator: register a Python function as an agent tool."""
        def _register(f):
            name = f.__name__
            self._tools[name] = f
            self._schemas[name] = schema or schema_from_function(f)
            return f
        return _register(fn) if fn else _register

    def register_many(self, *fns: Callable):
        for fn in fns:
            self.register(fn)

    # ── Lookup ───────────────────────────────────────────────────────────────

    def get_schema(self, name: str) -> dict:
        if name not in self._schemas:
            raise KeyError(f"Tool '{name}' not registered.")
        return self._schemas[name]

    def all_schemas(self) -> List[dict]:
        """Return all schemas — this is what you'd pass to the LLM."""
        return list(self._schemas.values())

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    # ── Execution ────────────────────────────────────────────────────────────

    def call(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool by name with the provided argument dict.
        Performs basic type coercion based on registered schema.
        """
        if name not in self._tools:
            raise KeyError(f"Unknown tool: '{name}'")
        fn = self._tools[name]
        schema = self._schemas[name]

        # Coerce types where possible
        props = schema["parameters"]["properties"]
        coerced: Dict[str, Any] = {}
        for k, v in arguments.items():
            expected = props.get(k, {}).get("type", "string")
            coerced[k] = _coerce(v, expected)

        return fn(**coerced)

    def __repr__(self):
        return f"ToolRegistry(tools={self.list_tools()})"


def _coerce(value: Any, json_type: str) -> Any:
    """Coerce a value to the expected JSON Schema type."""
    try:
        if json_type == "integer":
            return int(value)
        elif json_type == "number":
            return float(value)
        elif json_type == "boolean":
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes")
            return bool(value)
        elif json_type == "string":
            return str(value)
    except (ValueError, TypeError):
        pass
    return value


# ──────────────────────────────────────────────────────────────────────────────
# 3. ToolCall & ToolResult
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    """
    Represents a tool invocation requested by the model.
    Mirrors the Anthropic / OpenAI tool-use response format.
    """
    id: str             # Unique call ID (for matching results to calls)
    name: str           # Function name
    arguments: dict     # Parsed argument dict


@dataclass
class ToolResult:
    """
    Represents the outcome of a tool execution, to be fed back to the model.
    """
    call_id: str
    name: str
    content: Any        # Return value
    error: Optional[str] = None
    elapsed_ms: float = 0.0

    @property
    def is_error(self) -> bool:
        return self.error is not None

    def to_message(self) -> dict:
        """Format as a 'tool' role message for the context window."""
        if self.is_error:
            content = f"[ERROR] {self.error}"
        else:
            content = json.dumps(self.content) if not isinstance(self.content, str) else self.content
        return {
            "role": "tool",
            "tool_call_id": self.call_id,
            "name": self.name,
            "content": content,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 4. Dispatcher — executes one or many ToolCalls
# ──────────────────────────────────────────────────────────────────────────────

class ToolDispatcher:
    """
    Executes ToolCall objects against a ToolRegistry.

    Supports both sequential and (simulated) parallel execution.
    In production you'd use asyncio / ThreadPoolExecutor for true parallelism.
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.history: List[ToolResult] = []

    def execute(self, call: ToolCall) -> ToolResult:
        """Execute a single ToolCall and return the ToolResult."""
        t0 = time.perf_counter()
        try:
            content = self.registry.call(call.name, call.arguments)
            result = ToolResult(
                call_id=call.id,
                name=call.name,
                content=content,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:
            result = ToolResult(
                call_id=call.id,
                name=call.name,
                content=None,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        self.history.append(result)
        return result

    def execute_all(self, calls: List[ToolCall]) -> List[ToolResult]:
        """
        Execute a list of ToolCalls.

        All calls here run sequentially.  For parallel calls, tools with no
        data dependencies on each other can conceptually run in any order or
        concurrently — the model requests them in one shot.
        """
        return [self.execute(call) for call in calls]

    def clear_history(self):
        self.history.clear()

    def summary(self) -> str:
        lines = [f"{'Call ID':<20} {'Tool':<20} {'OK':<5} {'ms':>8}"]
        lines.append("-" * 56)
        for r in self.history:
            ok = "✓" if not r.is_error else "✗"
            lines.append(f"{r.call_id:<20} {r.name:<20} {ok:<5} {r.elapsed_ms:>8.1f}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Example Tools
# ──────────────────────────────────────────────────────────────────────────────

# Global registry for examples
TOOL_REGISTRY = ToolRegistry()


@TOOL_REGISTRY.register
def calculator(expression: str) -> float:
    """
    Evaluate a safe mathematical expression and return the numeric result.
    expression: A math expression, e.g. '2 + 3 * 4' or 'sqrt(16)'
    """
    # Only allow a safe subset of names
    safe_names = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    safe_names.update({"abs": abs, "round": round, "min": min, "max": max})
    result = eval(expression, {"__builtins__": {}}, safe_names)  # noqa: S307
    return float(result)


@TOOL_REGISTRY.register
def get_weather(location: str, units: str) -> dict:
    """
    Fetch current weather for a location (mocked).
    location: City name or 'City, Country'
    units: 'metric' or 'imperial'
    """
    # Mocked response — replace with real API call in production
    temp = 22.0 if units == "metric" else 71.6
    return {
        "location": location,
        "temperature": temp,
        "units": units,
        "condition": "Partly Cloudy",
        "humidity_pct": 58,
        "wind_kph": 14,
    }


@TOOL_REGISTRY.register
def web_search(query: str, max_results: int) -> list:
    """
    Search the web and return top results (mocked).
    query: Search query string
    max_results: Number of results to return (1-10)
    """
    max_results = max(1, min(int(max_results), 10))
    return [
        {
            "rank": i + 1,
            "title": f"Result {i+1} for: {query}",
            "url": f"https://example.com/result/{i+1}",
            "snippet": f"This is a mock search result snippet about '{query}'.",
        }
        for i in range(max_results)
    ]


@TOOL_REGISTRY.register
def read_file(path: str) -> str:
    """
    Read the contents of a local file.
    path: Absolute or relative file path
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")


@TOOL_REGISTRY.register
def python_eval(code: str) -> str:
    """
    Execute a Python code snippet and return stdout + the final expression.
    code: Valid Python code to run in a sandboxed namespace
    """
    import io, sys
    namespace: dict = {}
    stdout_capture = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_capture
    try:
        exec(compile(code, "<agent>", "exec"), namespace)   # noqa: S102
        output = stdout_capture.getvalue()
    except Exception:
        output = traceback.format_exc()
    finally:
        sys.stdout = old_stdout
    return output or "(no output)"


@TOOL_REGISTRY.register
def memory_lookup(query: str, top_k: int) -> list:
    """
    Retrieve relevant memories from the agent's semantic store (stub).
    query: Natural-language retrieval query
    top_k: Number of memories to return
    """
    # Stub — connect to SemanticMemory from 52_memory_types.py in real usage
    return [
        {"score": 0.95, "content": f"[stub] memory matching '{query}' (rank {i+1})"}
        for i in range(min(int(top_k), 3))
    ]


# ──────────────────────────────────────────────────────────────────────────────
# 6. Tool-call parsing (simulates model output → ToolCall list)
# ──────────────────────────────────────────────────────────────────────────────

def parse_tool_calls(raw: list) -> List[ToolCall]:
    """
    Parse the 'tool_calls' field of a model response into ToolCall objects.

    raw: List of dicts like:
        [{"id": "tc_1", "function": {"name": "add", "arguments": '{"a":1,"b":2}'}}]
    """
    calls = []
    for item in raw:
        fn = item.get("function", item)  # Handle both OpenAI and raw formats
        args_raw = fn.get("arguments", "{}")
        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        calls.append(ToolCall(
            id=item.get("id", f"tc_{len(calls)}"),
            name=fn["name"],
            arguments=args,
        ))
    return calls


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────

def demo():
    print("=" * 60)
    print("CHAPTER 50 — Tool Use Demo")
    print("=" * 60)

    print("\n[1] Registered tools:")
    for schema in TOOL_REGISTRY.all_schemas():
        params = list(schema["parameters"]["properties"].keys())
        print(f"  • {schema['name']}({', '.join(params)}) — {schema['description'][:60]}")

    dispatcher = ToolDispatcher(TOOL_REGISTRY)

    print("\n[2] Sequential tool calls:")
    calls = [
        ToolCall(id="tc_01", name="calculator",  arguments={"expression": "sqrt(144) + 2**8"}),
        ToolCall(id="tc_02", name="get_weather",  arguments={"location": "Tokyo", "units": "metric"}),
        ToolCall(id="tc_03", name="web_search",   arguments={"query": "LLM memory architectures", "max_results": 3}),
        ToolCall(id="tc_04", name="python_eval",  arguments={"code": "for i in range(3): print(f'step {i}')"}),
    ]

    results = dispatcher.execute_all(calls)
    for r in results:
        status = "OK" if not r.is_error else f"ERR: {r.error}"
        print(f"  [{r.call_id}] {r.name}: {status}")
        print(f"           → {r.content}")

    print("\n[3] Dispatcher summary:")
    print(dispatcher.summary())

    print("\n[4] Auto-generated schema for 'calculator':")
    print(json.dumps(TOOL_REGISTRY.get_schema("calculator"), indent=2))

    print("\n[5] Parallel tool call simulation (model requests multiple at once):")
    parallel_calls_raw = [
        {"id": "p_01", "function": {"name": "calculator", "arguments": '{"expression": "pi * 5**2"}'}},
        {"id": "p_02", "function": {"name": "get_weather", "arguments": '{"location": "Paris", "units": "metric"}'}},
    ]
    parsed = parse_tool_calls(parallel_calls_raw)
    parallel_results = dispatcher.execute_all(parsed)
    for r in parallel_results:
        print(f"  [{r.id_}] {r.name}: {r.content}" if hasattr(r, "id_") else f"  {r.name}: {r.content}")

    print("\nDone.")


if __name__ == "__main__":
    demo()
