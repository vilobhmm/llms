"""
Agentic Design Patterns — Chapter 51
=====================================
Six core patterns for building autonomous LLM agents:

  1. Chain-of-Thought (CoT)    — multi-step textual reasoning
  2. ReAct                     — Reason + Act + Observe loop
  3. Plan-and-Execute          — plan first, then run each step
  4. Reflection / Self-Critique— generate, critique, revise
  5. Tree-of-Thought (ToT)     — branch and score multiple paths
  6. Multi-Agent Supervisor    — supervisor delegates to specialist workers

All patterns accept a `model_fn: Callable[[list[dict]], str]` so they work
with any LLM (mock, OpenAI, Anthropic, local, etc.).

Run standalone:
    python 51_agentic_patterns.py
"""

from __future__ import annotations

import json
import random
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# Type alias: a model function takes a message list and returns a string
ModelFn = Callable[[List[Dict[str, str]]], str]


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _user(content: str) -> dict:
    return {"role": "user", "content": content}

def _assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}

def _system(content: str) -> dict:
    return {"role": "system", "content": content}

def _wrap(text: str, width: int = 80) -> str:
    return textwrap.fill(text, width=width)


# ──────────────────────────────────────────────────────────────────────────────
# PATTERN 1 — Chain-of-Thought (CoT)
# ──────────────────────────────────────────────────────────────────────────────

class ChainOfThought:
    """
    Prompts the model to reason step-by-step before giving its final answer.

    Approach:
      - Append "Think step by step." to the user prompt.
      - Optionally use few-shot CoT examples.
      - Parse final answer from "Answer:" marker.

    Best for: arithmetic, logical reasoning, multi-step deductions.
    """

    SYSTEM_PROMPT = (
        "You are a careful reasoner. Before answering any question, "
        "think through it step by step. End with 'Answer: <your answer>'."
    )

    def __init__(self, model_fn: ModelFn, few_shot_examples: List[Tuple[str, str]] = None):
        self.model_fn = model_fn
        self.few_shot = few_shot_examples or []

    def _build_messages(self, question: str) -> List[dict]:
        messages = [_system(self.SYSTEM_PROMPT)]
        for q, a in self.few_shot:
            messages.append(_user(q))
            messages.append(_assistant(a))
        messages.append(_user(f"{question}\n\nThink step by step."))
        return messages

    def run(self, question: str) -> dict:
        """
        Returns:
            {"question": ..., "reasoning": ..., "answer": ...}
        """
        messages = self._build_messages(question)
        raw = self.model_fn(messages)

        # Parse reasoning + answer
        if "Answer:" in raw:
            parts = raw.split("Answer:", 1)
            reasoning = parts[0].strip()
            answer = parts[1].strip()
        else:
            reasoning = raw.strip()
            answer = raw.strip().splitlines()[-1]

        return {"question": question, "reasoning": reasoning, "answer": answer}


# ──────────────────────────────────────────────────────────────────────────────
# PATTERN 2 — ReAct (Reason + Act + Observe)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ReActStep:
    thought: str
    action: Optional[str]
    action_input: Optional[str]
    observation: Optional[str]
    is_final: bool = False
    final_answer: Optional[str] = None


class ReActAgent:
    """
    The ReAct loop: Thought → Action → Observation → Thought → ...

    At each step the model outputs:
        Thought: <reasoning>
        Action: <tool_name>
        Action Input: <json args>

    The agent executes the action, feeds back:
        Observation: <result>

    Until the model outputs:
        Final Answer: <answer>

    Best for: question answering with external tools, web search, computation.
    """

    SYSTEM_PROMPT = textwrap.dedent("""
        You are a helpful assistant that answers questions using tools.
        At each turn output EXACTLY ONE of:

        Thought: <your internal reasoning>
        Action: <tool_name>
        Action Input: <JSON dict of arguments>

        OR when you are done:

        Final Answer: <your complete answer>

        Available tools:
        {tool_descriptions}
    """).strip()

    def __init__(
        self,
        model_fn: ModelFn,
        tools: Dict[str, Callable],        # name → callable(args_dict)
        max_steps: int = 10,
    ):
        self.model_fn = model_fn
        self.tools = tools
        self.max_steps = max_steps

    def _tool_desc(self) -> str:
        lines = []
        for name, fn in self.tools.items():
            doc_lines = (fn.__doc__ or "").splitlines()
            doc = doc_lines[0] if doc_lines else name
            lines.append(f"  • {name}: {doc}")
        return "\n".join(lines)

    def _parse(self, text: str) -> ReActStep:
        lines = text.strip().splitlines()
        thought = action = action_input = None
        is_final = False
        final_answer = None

        for line in lines:
            if line.startswith("Thought:"):
                thought = line[len("Thought:"):].strip()
            elif line.startswith("Action:"):
                action = line[len("Action:"):].strip()
            elif line.startswith("Action Input:"):
                action_input = line[len("Action Input:"):].strip()
            elif line.startswith("Final Answer:"):
                is_final = True
                final_answer = line[len("Final Answer:"):].strip()

        return ReActStep(
            thought=thought or text,
            action=action,
            action_input=action_input,
            observation=None,
            is_final=is_final,
            final_answer=final_answer,
        )

    def _execute_action(self, action: str, action_input: str) -> str:
        if action not in self.tools:
            return f"[Error] Unknown tool: '{action}'"
        try:
            args = json.loads(action_input) if action_input else {}
        except json.JSONDecodeError:
            args = {"input": action_input}
        try:
            result = self.tools[action](args)
            return str(result)
        except Exception as exc:
            return f"[Error] {type(exc).__name__}: {exc}"

    def run(self, task: str) -> dict:
        system = self.SYSTEM_PROMPT.format(tool_descriptions=self._tool_desc())
        messages = [_system(system), _user(task)]
        steps: List[ReActStep] = []

        for _ in range(self.max_steps):
            raw = self.model_fn(messages)
            step = self._parse(raw)
            messages.append(_assistant(raw))

            if step.is_final:
                steps.append(step)
                break

            # Execute action
            if step.action:
                obs = self._execute_action(step.action, step.action_input or "{}")
                step.observation = obs
                observation_msg = f"Observation: {obs}"
                messages.append(_user(observation_msg))

            steps.append(step)
        else:
            step = ReActStep(thought="Max steps reached", action=None,
                             action_input=None, observation=None,
                             is_final=True, final_answer="Max steps exceeded.")
            steps.append(step)

        return {"task": task, "steps": steps, "answer": steps[-1].final_answer}


# ──────────────────────────────────────────────────────────────────────────────
# PATTERN 3 — Plan-and-Execute
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ExecutionStep:
    index: int
    description: str
    result: Optional[str] = None
    status: str = "pending"   # pending | done | failed


class PlanAndExecute:
    """
    Two-phase agent:
      Phase 1 — Planner: Break the task into ordered steps (JSON list).
      Phase 2 — Executor: Run each step using tools, updating the plan.

    The executor can re-plan if a step fails or produces unexpected results.

    Best for: complex multi-step tasks, file operations, long research tasks.
    """

    PLANNER_SYSTEM = (
        "You are a task planner. Given a goal, output a JSON list of steps "
        "to accomplish it. Each step is a string. Output only valid JSON.\n"
        'Example: ["Step 1: ...", "Step 2: ...", "Step 3: ..."]'
    )

    EXECUTOR_SYSTEM = (
        "You are a task executor. Given a step description and the results of "
        "previous steps, output the result of this step. Be concise."
    )

    def __init__(
        self,
        model_fn: ModelFn,
        executor_tools: Dict[str, Callable] = None,
        max_replan: int = 2,
    ):
        self.model_fn = model_fn
        self.executor_tools = executor_tools or {}
        self.max_replan = max_replan

    def _plan(self, goal: str, previous_results: list = None) -> List[str]:
        context = ""
        if previous_results:
            context = "\nPrevious results:\n" + "\n".join(
                f"  Step {i+1}: {r}" for i, r in enumerate(previous_results)
            )
        messages = [
            _system(self.PLANNER_SYSTEM),
            _user(f"Goal: {goal}{context}\n\nOutput a JSON list of steps."),
        ]
        raw = self.model_fn(messages)
        # Extract JSON
        try:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            return json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            # Fallback: split by newline
            return [l.strip() for l in raw.splitlines() if l.strip()]

    def _execute_step(self, step: str, context: str) -> str:
        messages = [
            _system(self.EXECUTOR_SYSTEM),
            _user(f"Context so far:\n{context}\n\nExecute this step: {step}"),
        ]
        return self.model_fn(messages)

    def run(self, goal: str) -> dict:
        plan_steps = self._plan(goal)
        exec_steps = [ExecutionStep(i, desc) for i, desc in enumerate(plan_steps)]
        context_lines = []
        replan_count = 0

        i = 0
        while i < len(exec_steps):
            step = exec_steps[i]
            step.status = "running"
            context = "\n".join(context_lines)

            try:
                result = self._execute_step(step.description, context)
                step.result = result
                step.status = "done"
                context_lines.append(f"Step {i+1} ({step.description}): {result}")
            except Exception as exc:
                step.result = str(exc)
                step.status = "failed"
                if replan_count < self.max_replan:
                    replan_count += 1
                    new_steps = self._plan(goal, context_lines)
                    exec_steps = exec_steps[:i+1] + [
                        ExecutionStep(j + i + 1, d) for j, d in enumerate(new_steps[i+1:])
                    ]
            i += 1

        return {
            "goal": goal,
            "steps": exec_steps,
            "final_result": exec_steps[-1].result if exec_steps else None,
        }


# ──────────────────────────────────────────────────────────────────────────────
# PATTERN 4 — Reflection / Self-Critique
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ReflectionRound:
    draft: str
    critique: str
    revised: str


class ReflectionAgent:
    """
    Three-stage generate → critique → revise loop.

    Stage 1 — Draft: Generate an initial response.
    Stage 2 — Critique: Self-evaluate the draft (factual, logical, quality).
    Stage 3 — Revise: Improve the draft based on the critique.

    Can run multiple reflection rounds for iterative improvement.

    Best for: writing tasks, code generation, answer quality improvement.
    """

    DRAFT_SYSTEM    = "You are a helpful assistant. Answer the user's question."
    CRITIQUE_SYSTEM = (
        "You are a critical reviewer. Given a draft answer, identify its "
        "weaknesses: missing info, factual errors, poor clarity, or logic gaps. "
        "Be specific and constructive."
    )
    REVISE_SYSTEM   = (
        "You are a skilled editor. Given a draft answer and a critique, "
        "produce an improved version that addresses all critiques."
    )

    def __init__(self, model_fn: ModelFn, num_rounds: int = 2):
        self.model_fn = model_fn
        self.num_rounds = max(1, num_rounds)

    def run(self, question: str) -> dict:
        # Stage 1: Draft
        draft = self.model_fn([_system(self.DRAFT_SYSTEM), _user(question)])
        rounds: List[ReflectionRound] = []

        current = draft
        for _ in range(self.num_rounds):
            # Stage 2: Critique
            critique = self.model_fn([
                _system(self.CRITIQUE_SYSTEM),
                _user(f"Question: {question}\n\nDraft answer:\n{current}"),
            ])
            # Stage 3: Revise
            revised = self.model_fn([
                _system(self.REVISE_SYSTEM),
                _user(f"Question: {question}\n\nDraft:\n{current}\n\nCritique:\n{critique}"),
            ])
            rounds.append(ReflectionRound(draft=current, critique=critique, revised=revised))
            current = revised

        return {"question": question, "initial_draft": draft, "rounds": rounds, "final": current}


# ──────────────────────────────────────────────────────────────────────────────
# PATTERN 5 — Tree of Thought (ToT)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ThoughtNode:
    thought: str
    score: float
    depth: int
    children: List["ThoughtNode"] = field(default_factory=list)
    parent: Optional["ThoughtNode"] = None

    def path(self) -> List[str]:
        """Return ordered list of thoughts from root to this node."""
        nodes = []
        node = self
        while node:
            nodes.append(node.thought)
            node = node.parent
        return list(reversed(nodes))


class TreeOfThought:
    """
    Breadth-first search over a tree of thoughts.

    At each depth:
      - Generate N candidate next thoughts from each current node
      - Score each candidate with the evaluator
      - Keep top-K branches for the next level
      - Stop when any branch reaches a final answer

    Best for: planning, puzzles, strategy problems with multiple approaches.
    """

    def __init__(
        self,
        model_fn: ModelFn,
        n_branches: int = 3,
        beam_width: int = 2,
        max_depth: int = 4,
    ):
        self.model_fn = model_fn
        self.n_branches = n_branches
        self.beam_width = beam_width
        self.max_depth = max_depth

    def _generate_thoughts(self, problem: str, path: List[str]) -> List[str]:
        context = "\n".join(f"Step {i+1}: {t}" for i, t in enumerate(path))
        prompt = (
            f"Problem: {problem}\n\n"
            f"Reasoning so far:\n{context}\n\n"
            f"Generate {self.n_branches} different next reasoning steps. "
            f"Output as a JSON list of strings."
        )
        raw = self.model_fn([_user(prompt)])
        try:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            return json.loads(raw[start:end])[:self.n_branches]
        except (ValueError, json.JSONDecodeError):
            return [raw.strip()]

    def _score_thought(self, problem: str, path: List[str], thought: str) -> float:
        context = "\n".join(path + [thought])
        prompt = (
            f"Problem: {problem}\n\nReasoning path:\n{context}\n\n"
            "Score this reasoning path's likelihood of leading to the correct answer. "
            "Output ONLY a number from 0.0 (worst) to 1.0 (best)."
        )
        raw = self.model_fn([_user(prompt)])
        try:
            return float("".join(c for c in raw if c.isdigit() or c == "."))
        except ValueError:
            return random.uniform(0.3, 0.7)

    def _is_final(self, thought: str) -> bool:
        keywords = ["answer:", "solution:", "therefore:", "conclusion:", "final answer"]
        return any(kw in thought.lower() for kw in keywords)

    def run(self, problem: str) -> dict:
        root = ThoughtNode(thought="[Start]", score=1.0, depth=0)
        beam = [root]
        best_node: Optional[ThoughtNode] = None

        for depth in range(1, self.max_depth + 1):
            candidates: List[ThoughtNode] = []

            for node in beam:
                thoughts = self._generate_thoughts(problem, node.path()[1:])
                for thought in thoughts:
                    score = self._score_thought(problem, node.path()[1:], thought)
                    child = ThoughtNode(thought=thought, score=score, depth=depth, parent=node)
                    node.children.append(child)
                    candidates.append(child)

                    if self._is_final(thought):
                        best_node = child
                        break
                if best_node:
                    break
            if best_node:
                break

            # Keep top-K by score
            beam = sorted(candidates, key=lambda n: n.score, reverse=True)[:self.beam_width]

        if not best_node and beam:
            best_node = beam[0]

        return {
            "problem": problem,
            "best_path": best_node.path() if best_node else [],
            "final_score": best_node.score if best_node else 0.0,
            "depth_reached": best_node.depth if best_node else 0,
            "answer": best_node.thought if best_node else "No solution found.",
        }


# ──────────────────────────────────────────────────────────────────────────────
# PATTERN 6 — Multi-Agent Supervisor
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class WorkerSpec:
    name: str
    system_prompt: str
    description: str


@dataclass
class AgentMessage:
    sender: str
    recipient: str
    content: str


class MultiAgentSupervisor:
    """
    Supervisor ↔ Worker multi-agent pattern.

    The Supervisor:
      1. Receives a task
      2. Decides which worker to delegate to (or answers directly)
      3. Collects the worker's output
      4. Decides to delegate again or synthesize a final answer

    Workers:
      Each has a specialized system prompt (researcher, coder, critic, etc.)

    Communication is modeled as a message-passing log for full observability.

    Best for: tasks requiring different specializations, parallel research.
    """

    SUPERVISOR_SYSTEM = textwrap.dedent("""
        You are a supervisor managing a team of specialist agents.
        Available workers:
        {worker_list}

        At each turn, output EXACTLY one of:
          DELEGATE TO <worker_name>: <task for the worker>
          FINAL ANSWER: <your synthesized answer>

        Use DELEGATE to gather information or subtask. Use FINAL ANSWER when done.
    """).strip()

    def __init__(
        self,
        model_fn: ModelFn,
        workers: List[WorkerSpec],
        max_turns: int = 8,
    ):
        self.model_fn = model_fn
        self.workers = {w.name: w for w in workers}
        self.max_turns = max_turns

    def _supervisor_system(self) -> str:
        worker_list = "\n".join(
            f"  • {w.name}: {w.description}" for w in self.workers.values()
        )
        return self.SUPERVISOR_SYSTEM.format(worker_list=worker_list)

    def _run_worker(self, worker_name: str, task: str) -> str:
        worker = self.workers[worker_name]
        return self.model_fn([_system(worker.system_prompt), _user(task)])

    def run(self, task: str) -> dict:
        messages = [_system(self._supervisor_system()), _user(task)]
        log: List[AgentMessage] = [AgentMessage("user", "supervisor", task)]
        final_answer: Optional[str] = None

        for _ in range(self.max_turns):
            raw = self.model_fn(messages)
            messages.append(_assistant(raw))

            if raw.strip().upper().startswith("FINAL ANSWER"):
                final_answer = raw.split(":", 1)[-1].strip()
                log.append(AgentMessage("supervisor", "user", final_answer))
                break
            elif "DELEGATE TO" in raw.upper():
                # Parse: "DELEGATE TO researcher: find info about X"
                remainder = raw.split("DELEGATE TO", 1)[-1].strip()
                if ":" in remainder:
                    worker_name, subtask = remainder.split(":", 1)
                    worker_name = worker_name.strip()
                    subtask = subtask.strip()
                else:
                    worker_name = remainder.split()[0]
                    subtask = task

                log.append(AgentMessage("supervisor", worker_name, subtask))

                if worker_name in self.workers:
                    worker_result = self._run_worker(worker_name, subtask)
                    log.append(AgentMessage(worker_name, "supervisor", worker_result))
                    observation = f"[{worker_name} result]: {worker_result}"
                    messages.append(_user(observation))
                else:
                    err = f"Worker '{worker_name}' not found."
                    messages.append(_user(f"[Error] {err}"))
                    log.append(AgentMessage("system", "supervisor", err))
            else:
                final_answer = raw.strip()
                log.append(AgentMessage("supervisor", "user", final_answer))
                break

        return {"task": task, "log": log, "final_answer": final_answer or "No answer produced."}


# ──────────────────────────────────────────────────────────────────────────────
# Mock model for demonstration
# ──────────────────────────────────────────────────────────────────────────────

def mock_model(messages: List[dict]) -> str:
    """Simple mock model that returns canned responses based on keywords."""
    last = messages[-1]["content"].lower() if messages else ""

    if "step by step" in last:
        return ("Thought: Let me reason through this carefully.\n"
                "Step 1: Identify the key information.\n"
                "Step 2: Apply the relevant logic.\n"
                "Step 3: Arrive at the conclusion.\n"
                "Answer: 42 (mock answer)")
    if "thought:" in last or "action:" in last:
        return "Final Answer: This is the final mock answer after using tools."
    if "json list" in last and "steps" in last:
        return '["Step 1: Gather information", "Step 2: Analyze data", "Step 3: Summarize findings"]'
    if "json list" in last and "reasoning" in last:
        return '["Approach A: try brute force", "Approach B: use dynamic programming", "Approach C: greedy"]'
    if "0.0" in last and "1.0" in last:
        return "0.75"
    if "critique" in last or "weaknesses" in last:
        return "The draft lacks specific examples and could be more concise."
    if "improved version" in last:
        return "Improved answer with concrete examples and clearer structure."
    if "delegate" in last.upper() or "worker" in last:
        return "FINAL ANSWER: Synthesized answer from all workers."
    return f"Mock response for: {messages[-1]['content'][:60]}..."


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────

def demo():
    print("=" * 60)
    print("CHAPTER 51 — Agentic Design Patterns Demo")
    print("=" * 60)
    task = "What is the capital of France, and what is its population?"

    print("\n[1] Chain-of-Thought")
    cot = ChainOfThought(mock_model)
    result = cot.run(task)
    print(f"  Question:  {result['question']}")
    print(f"  Reasoning: {result['reasoning'][:80]}...")
    print(f"  Answer:    {result['answer']}")

    print("\n[2] ReAct")
    tools = {
        "search": lambda args: f"Search results for '{args.get('query', '')}'",
        "calculate": lambda args: str(eval(args.get("expr", "0"), {}, {})),  # noqa: S307
    }
    react = ReActAgent(mock_model, tools, max_steps=3)
    result = react.run(task)
    print(f"  Steps: {len(result['steps'])}")
    print(f"  Answer: {result['answer']}")

    print("\n[3] Plan-and-Execute")
    pe = PlanAndExecute(mock_model)
    result = pe.run("Research and write a summary of LLM memory architectures.")
    print(f"  Goal: {result['goal']}")
    for step in result["steps"]:
        print(f"  Step {step.index+1} [{step.status}]: {step.description[:50]}")

    print("\n[4] Reflection")
    ra = ReflectionAgent(mock_model, num_rounds=2)
    result = ra.run("Explain what an LLM is in simple terms.")
    print(f"  Rounds: {len(result['rounds'])}")
    print(f"  Final: {result['final'][:100]}...")

    print("\n[5] Tree of Thought")
    tot = TreeOfThought(mock_model, n_branches=2, beam_width=2, max_depth=2)
    result = tot.run("Find the best algorithm for sorting a very large list.")
    print(f"  Depth reached: {result['depth_reached']}")
    print(f"  Best path: {len(result['best_path'])} steps")
    print(f"  Answer: {result['answer'][:80]}...")

    print("\n[6] Multi-Agent Supervisor")
    workers = [
        WorkerSpec("researcher", "You are a research specialist. Find facts.", "Researches topics"),
        WorkerSpec("analyst",    "You are a data analyst. Analyze findings.",  "Analyzes data"),
        WorkerSpec("writer",     "You are a technical writer. Draft reports.",  "Writes summaries"),
    ]
    supervisor = MultiAgentSupervisor(mock_model, workers, max_turns=4)
    result = supervisor.run("Produce a research summary on transformer architectures.")
    print(f"  Task: {result['task']}")
    print(f"  Messages exchanged: {len(result['log'])}")
    print(f"  Final: {result['final_answer'][:100]}...")

    print("\nAll patterns completed successfully.")


if __name__ == "__main__":
    demo()
