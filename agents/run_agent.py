"""
run_agent.py — Interactive Agent Demo
======================================
Entry point for the agents/ module. Runs individual chapter demos
or launches an interactive REPL with the orchestrated agent.

Usage:
    # Run all chapter demos sequentially
    python run_agent.py

    # Run a specific chapter demo
    python run_agent.py --chapter 50
    python run_agent.py --chapter 51
    python run_agent.py --chapter 52
    python run_agent.py --chapter 53
    python run_agent.py --chapter 54
    python run_agent.py --chapter 55

    # Interactive REPL with the full orchestrated agent
    python run_agent.py --interactive

    # Interactive with a specific agentic pattern
    python run_agent.py --interactive --pattern cot
    python run_agent.py --interactive --pattern react
    python run_agent.py --interactive --pattern plan
    python run_agent.py --interactive --pattern reflect

    # Debug mode: print context window + memory state each turn
    python run_agent.py --interactive --debug

    # Use real Claude API (requires ANTHROPIC_API_KEY env var)
    python run_agent.py --interactive --use-claude
"""

import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(__file__)


def _load(filename: str):
    """Load a digit-prefixed module file by filename."""
    path = os.path.join(HERE, filename)
    spec = importlib.util.spec_from_file_location(filename, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[filename] = mod   # register before exec so dataclasses work
    spec.loader.exec_module(mod)
    return mod


CHAPTERS = {
    50: "50_tool_use.py",
    51: "51_agentic_patterns.py",
    52: "52_memory_types.py",
    53: "53_memory_store.py",
    54: "54_context_debugger.py",
    55: "55_agent_orchestrator.py",
}

CHAPTER_TITLES = {
    50: "Tool Use & Function Calling",
    51: "Agentic Design Patterns (CoT / ReAct / Plan / Reflect / ToT / Multi-Agent)",
    52: "Memory Types (Working / Episodic / Semantic / Procedural)",
    53: "Memory Stores (Vector / Sliding Window / Episodic / KV / Hierarchical)",
    54: "Context Debugger (Snapshot / Visualizer / Diff / Inspector)",
    55: "Agent Orchestrator (Full Pipeline)",
}


def run_chapter(chapter: int):
    """Run the demo() function for a specific chapter."""
    filename = CHAPTERS.get(chapter)
    if not filename:
        print(f"[Error] Chapter {chapter} not found. Valid: {list(CHAPTERS.keys())}")
        sys.exit(1)

    title = CHAPTER_TITLES.get(chapter, filename)
    print(f"\n{'#' * 70}")
    print(f"# Chapter {chapter}: {title}")
    print(f"{'#' * 70}\n")

    mod = _load(filename)
    if hasattr(mod, "demo"):
        mod.demo()
    else:
        print(f"[Warning] No demo() function found in {filename}")


def run_all_chapters():
    """Run all chapter demos in sequence."""
    print("\n" + "=" * 70)
    print(" AGENTS MODULE — Full Demo Run")
    print("=" * 70)
    print("\nChapters:")
    for num, title in CHAPTER_TITLES.items():
        print(f"  {num}: {title}")
    print()

    for chapter in sorted(CHAPTERS.keys()):
        run_chapter(chapter)
        print("\n" + "─" * 70)

    print("\n✓ All chapter demos completed.")


def run_interactive(pattern: str = "react", debug: bool = False, use_claude: bool = False):
    """Launch an interactive REPL with the orchestrated agent."""
    orchestrator = _load("55_agent_orchestrator.py")

    print("\n" + "=" * 70)
    print(" INTERACTIVE AGENT REPL")
    print("=" * 70)
    print(f" Pattern: {pattern}")
    print(f" Debug:   {debug}")
    print(f" Model:   {'Claude API' if use_claude else 'Mock (set --use-claude for real LLM)'}")
    print("\nCommands:")
    print("  /debug     — Print context window snapshot")
    print("  /memory    — Print memory state")
    print("  /tools     — List registered tools")
    print("  /summary   — Print session summary")
    print("  /clear     — Clear working memory")
    print("  /quit      — Exit")
    print("=" * 70 + "\n")

    # Build model function
    if use_claude:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("[Error] Set ANTHROPIC_API_KEY environment variable.")
            sys.exit(1)
        model_fn = orchestrator.make_claude_fn(api_key=api_key)
        print("[Claude API connected]\n")
    else:
        model_fn = None  # Will use mock_model

    # Build agent
    agent = orchestrator.build_agent(
        model_fn=model_fn,
        pattern=pattern,
        system_prompt=(
            "You are a helpful AI agent with tools for web search, calculations, "
            "and file reading. Use tools when appropriate. Be concise and accurate."
        ),
        debug=debug,
        consolidate_every=5,
    )

    # Seed some knowledge
    agent.memory.semantic.add(
        "This is an educational implementation of LLM agents from scratch.",
        source="system",
        confidence=1.0,
    )

    # REPL loop
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        # Commands
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ("/quit", "/exit", "/q"):
                print("Goodbye!")
                break
            elif cmd == "/debug":
                agent.debug_context()
            elif cmd == "/memory":
                agent.debug_memory()
            elif cmd == "/tools":
                agent.debug_tools()
            elif cmd == "/summary":
                print(agent.session_summary())
            elif cmd == "/clear":
                agent.memory.working.clear()
                print("[Working memory cleared]")
            else:
                print(f"Unknown command: {user_input}")
            continue

        # Normal chat turn
        response = agent.chat(user_input)
        print(f"\nAgent: {response}")


def main():
    parser = argparse.ArgumentParser(
        description="Run agent chapter demos or interactive REPL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--chapter", "-c", type=int, choices=list(CHAPTERS.keys()),
        help="Run demo for a specific chapter (50-55)",
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Launch interactive REPL",
    )
    parser.add_argument(
        "--pattern", "-p",
        choices=["direct", "cot", "react", "plan", "reflect"],
        default="react",
        help="Agentic pattern for interactive mode (default: react)",
    )
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Enable context/memory debug output each turn",
    )
    parser.add_argument(
        "--use-claude", action="store_true",
        help="Use real Claude API (requires ANTHROPIC_API_KEY env var)",
    )

    args = parser.parse_args()

    if args.chapter:
        run_chapter(args.chapter)
    elif args.interactive:
        run_interactive(pattern=args.pattern, debug=args.debug, use_claude=args.use_claude)
    else:
        run_all_chapters()


if __name__ == "__main__":
    main()
