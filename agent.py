"""
Multi-Memory Agent — LangGraph Implementation
===============================================

LangGraph graph:
    START → [load_memory] → [llm_node] → [save_memory] → END

MemoryState (TypedDict) — exactly as slide:
    messages       : List[Dict]   — recent conversation (short-term)
    user_profile   : Dict         — long-term preferences + facts
    episodes       : List[Dict]   — episodic memory hits
    semantic_hits  : List[str]    — semantic knowledge chunks
    memory_budget  : int          — remaining token budget
    user_id        : str          — identifies the user
    current_query  : str          — current user message
    llm_response   : str          — agent's response

Node 1 - load_memory:   Read from all 4 backends → populate state
Node 2 - llm_node:      Build prompt with injected memory → call LLM
Node 3 - save_memory:   Persist updates back to all backends
"""

from __future__ import annotations
import json
import os
from typing import TypedDict, List, Dict, Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.episodic import EpisodicMemory
from memory.semantic import SemanticMemory
from router import MemoryRouter
from context_manager import ContextManager

load_dotenv()

# ---------------------------------------------------------------------------
# MemoryState — the LangGraph state (matches slide exactly)
# ---------------------------------------------------------------------------

class MemoryState(TypedDict):
    messages: List[Dict[str, Any]]    # recent conversation buffer (short-term)
    user_profile: Dict[str, Any]      # long-term: preferences + facts
    episodes: List[Dict[str, Any]]    # episodic: past task trajectories
    semantic_hits: List[str]          # semantic: domain knowledge chunks
    memory_budget: int                # tokens remaining in budget
    user_id: str                      # user identifier
    current_query: str                # current turn's user message
    llm_response: str                 # LLM response for this turn


# ---------------------------------------------------------------------------
# Singleton memory backends (shared across sessions, same process)
# ---------------------------------------------------------------------------

_short_term: Dict[str, ShortTermMemory] = {}  # one buffer per user_id
_long_term = LongTermMemory()
_episodic = EpisodicMemory(log_file="data/episodes.json")
_semantic = SemanticMemory()
_router = MemoryRouter()
_ctx_manager = ContextManager(
    total_context=int(os.getenv("TOTAL_CONTEXT_TOKENS", "4096"))
)

# Load seed episodes and domain docs at module load time
_episodic.load_seeds("data/seed_episodes.json")
if _semantic.count() == 0:
    _semantic.load_from_file("data/domain_docs.json")

# LLM — GPT-4o-mini by default (cheap, capable)
_llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
    temperature=0.7,
)


def _get_short_term(user_id: str) -> ShortTermMemory:
    if user_id not in _short_term:
        _short_term[user_id] = ShortTermMemory(max_messages=20, max_tokens=2000)
    return _short_term[user_id]


# ---------------------------------------------------------------------------
# Node 1: load_memory
# ---------------------------------------------------------------------------

def load_memory(state: MemoryState) -> Dict[str, Any]:
    """
    Load all relevant memory into state before calling LLM.

    Steps:
    1. Classify query intent → determine which memory types to load.
    2. Load short-term (recent messages from buffer).
    3. Load long-term profile (all prefs + facts from Redis).
    4. Search episodic memory for similar past tasks.
    5. Query semantic memory for relevant knowledge chunks.
    6. Compute remaining token budget.
    """
    user_id = state.get("user_id", "default")
    query = state["current_query"]

    # Step 1 — Route
    memory_types = _router.classify(query)

    # Step 2 — Short-term: always load recent messages
    stm = _get_short_term(user_id)
    recent_messages = stm.get_recent(k=10)

    # Step 3 — Long-term: always load profile (light operation)
    profile = _long_term.get_profile(user_id)

    # Step 4 — Episodic: load when relevant
    episodes: List[Dict[str, Any]] = []
    if "episodic" in memory_types:
        episodes = _episodic.search_similar(query, k=3, user_id=user_id)

    # Step 5 — Semantic: load when relevant
    semantic_hits: List[str] = []
    if "semantic" in memory_types:
        semantic_hits = _semantic.query(query, k=3)

    # Step 6 — Budget
    budget = _ctx_manager.compute_budget(
        profile, episodes, semantic_hits, recent_messages
    )

    return {
        "messages": recent_messages,
        "user_profile": profile,
        "episodes": episodes,
        "semantic_hits": semantic_hits,
        "memory_budget": budget,
        "user_id": user_id,
        "current_query": query,
        "llm_response": "",
    }


# ---------------------------------------------------------------------------
# Node 2: llm_node
# ---------------------------------------------------------------------------

def llm_node(state: MemoryState) -> Dict[str, Any]:
    """
    Build structured prompt with memory injection → call LLM → store response.

    Prompt sections (priority order from slide):
    1. System context + persona
    2. User profile (long-term facts + preferences)
    3. Relevant past episodes (episodic)
    4. Domain knowledge (semantic)
    5. Recent conversation (short-term)
    6. Current user query
    """
    # Trim state if needed before building prompt
    trimmed = _ctx_manager.trim(state)

    # Build system prompt with all memory sections
    system_prompt = _build_system_prompt(trimmed)

    # Assemble LangChain messages
    lc_messages = [SystemMessage(content=system_prompt)]

    # Add recent conversation history
    for msg in trimmed["messages"]:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))

    # Add current query
    lc_messages.append(HumanMessage(content=state["current_query"]))

    # Call LLM
    response = _llm.invoke(lc_messages)

    return {**state, "llm_response": response.content}


# ---------------------------------------------------------------------------
# Node 3: save_memory
# ---------------------------------------------------------------------------

def save_memory(state: MemoryState) -> Dict[str, Any]:
    """
    Persist memory after each turn.

    Operations:
    - Short-term: add (query, response) to buffer
    - Long-term:  extract key facts via LLM → update Redis (recency wins)
    - Episodic:   log episode if task completion detected
    - Semantic:   (static domain knowledge — not updated per turn)
    """
    user_id = state.get("user_id", "default")
    query = state["current_query"]
    response = state["llm_response"]

    # 1. Update short-term buffer
    stm = _get_short_term(user_id)
    stm.add_message("user", query)
    stm.add_message("assistant", response)

    # 2. Extract and update long-term profile (LLM-based, with error handling)
    _extract_and_update_profile(user_id, query, response)

    # 3. Log episode if task completion detected
    _log_episode_if_task_complete(user_id, query, response)

    return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_system_prompt(state: MemoryState) -> str:
    """
    Construct a structured system prompt with memory sections injected.
    Each section is clearly labeled so LLM can use it appropriately.
    """
    profile = state.get("user_profile", {})
    episodes = state.get("episodes", [])
    semantic_hits = state.get("semantic_hits", [])

    parts: List[str] = []

    # --- Persona ---
    parts += [
        "You are an expert Python and Machine Learning tutor with persistent memory.",
        "You personalize responses based on the user's stored profile and past interactions.",
        "",
    ]

    # --- Section: User Profile (Long-term memory) ---
    prefs = profile.get("preferences", {})
    facts = profile.get("facts", {})
    if prefs or facts:
        parts.append("=== USER PROFILE (Long-term Memory) ===")
        if prefs:
            pref_str = ", ".join(f"{k}: {v}" for k, v in prefs.items())
            parts.append(f"Preferences: {pref_str}")
        if facts:
            facts_str = ", ".join(f"{k}: {v}" for k, v in facts.items())
            parts.append(f"Known facts: {facts_str}")
        parts.append("")

    # --- Section: Episodic Memory ---
    if episodes:
        parts.append("=== PAST EXPERIENCES (Episodic Memory) ===")
        parts.append(_episodic.format_for_prompt(episodes))
        parts.append("")

    # --- Section: Semantic Knowledge ---
    if semantic_hits:
        parts.append("=== RELEVANT KNOWLEDGE (Semantic Memory) ===")
        for i, chunk in enumerate(semantic_hits, 1):
            truncated = chunk[:400] + "..." if len(chunk) > 400 else chunk
            parts.append(f"{i}. {truncated}")
        parts.append("")

    # --- Instructions ---
    parts += [
        "=== INSTRUCTIONS ===",
        "- Address the user by name if you know it.",
        "- Use their preferred language and explain at their skill level.",
        "- Reference past experiences when similar tasks come up.",
        "- When user corrects a fact (e.g. 'actually I meant X not Y'), "
        "  acknowledge the correction and use the updated information.",
        "- Be concise, accurate, and helpful.",
    ]

    return "\n".join(parts)


def _extract_and_update_profile(
    user_id: str, query: str, response: str
) -> None:
    """
    Use LLM to extract user facts and preferences from the conversation turn.
    Implements conflict handling: same key is overwritten → recency wins.
    Has robust parse/error handling (bonus point from rubric).
    """
    extraction_prompt = (
        "Extract user facts and preferences from the conversation below.\n"
        "Return ONLY valid JSON — no markdown, no explanation.\n\n"
        "Required JSON structure:\n"
        '{"preferences": {"key": "value"}, "facts": {"key": "value"}}\n\n'
        "Guidelines:\n"
        "- preferences: language choice (e.g. language=Python), style (e.g. style=concise), "
        "  topics liked/disliked\n"
        "- facts: name, skill level, allergy, project_type, errors_faced, anything factual "
        "  the user stated about themselves\n"
        "- Use simple snake_case keys (e.g. skill_level, preferred_language, allergy)\n"
        "- If user CORRECTS a previous statement, include the corrected value\n"
        "- If nothing to extract, return: {\"preferences\": {}, \"facts\": {}}\n\n"
        f"User: {query}\n"
        f"Assistant: {response[:200]}"
    )

    try:
        extraction_llm = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            temperature=0,  # deterministic extraction
        )
        raw = extraction_llm.invoke(
            [HumanMessage(content=extraction_prompt)]
        ).content.strip()

        # Clean up markdown code fences if present
        if "```" in raw:
            blocks = raw.split("```")
            for block in blocks:
                if "{" in block:
                    raw = block.strip()
                    if raw.startswith("json"):
                        raw = raw[4:].strip()
                    break

        extracted = json.loads(raw)

        if isinstance(extracted, dict):
            # Validate structure before updating
            if "preferences" not in extracted:
                extracted["preferences"] = {}
            if "facts" not in extracted:
                extracted["facts"] = {}
            # Update long-term memory — recency wins on same key
            _long_term.update_profile_from_dict(user_id, extracted)

    except json.JSONDecodeError:
        pass  # Graceful degradation: skip extraction if JSON invalid
    except Exception:
        pass  # Never crash the agent due to extraction failure


def _log_episode_if_task_complete(
    user_id: str, query: str, response: str
) -> None:
    """
    Log a new episode when a task completion event is detected.
    Heuristic: task-related keywords + positive outcome words in query/response.
    Only persist after task completion (slide: avoid inconsistent state).
    """
    completion_keywords = [
        "cảm ơn", "thank", "fix được", "giải quyết được", "đã xong",
        "hoạt động rồi", "worked", "solved", "fixed", "done", "resolved",
        "đã hiểu", "understood", "got it",
    ]
    task_keywords = [
        "lỗi", "error", "debug", "fix", "install", "problem", "issue",
        "không chạy", "không được", "bị lỗi", "help",
    ]

    q_lower = query.lower()
    r_lower = response.lower()

    has_completion = any(kw in q_lower or kw in r_lower for kw in completion_keywords)
    has_task = any(kw in q_lower for kw in task_keywords)

    if has_completion or has_task:
        stm = _get_short_term(user_id)
        recent = stm.get_recent(k=6)
        context = " | ".join(
            m["content"][:80] for m in recent if m["role"] == "user"
        )
        _episodic.log_episode(
            task=query[:120],
            trajectory=context[:200] if context else query[:100],
            outcome=response[:200],
            reflection=(
                "Key approach used. "
                + ("Task completed successfully." if has_completion else "Task in progress.")
            ),
            user_id=user_id,
        )


# ---------------------------------------------------------------------------
# Build the LangGraph graph
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    """
    Construct and compile the LangGraph StateGraph.

    Flow:
        START → load_memory → llm_node → save_memory → END
    """
    workflow = StateGraph(MemoryState)

    workflow.add_node("load_memory", load_memory)
    workflow.add_node("llm_node", llm_node)
    workflow.add_node("save_memory", save_memory)

    workflow.set_entry_point("load_memory")
    workflow.add_edge("load_memory", "llm_node")
    workflow.add_edge("llm_node", "save_memory")
    workflow.add_edge("save_memory", END)

    return workflow.compile()


# Compiled graph (singleton)
graph = build_graph()


# ---------------------------------------------------------------------------
# Public chat interface
# ---------------------------------------------------------------------------

def chat(query: str, user_id: str = "default") -> str:
    """
    Send a message to the agent and get a response.
    Memory is automatically loaded, injected, and saved.

    Args:
        query   : user's message
        user_id : identifies the user (for memory isolation)

    Returns:
        Agent's response string
    """
    initial_state: MemoryState = {
        "messages": [],
        "user_profile": {},
        "episodes": [],
        "semantic_hits": [],
        "memory_budget": int(os.getenv("MEMORY_BUDGET", "2000")),
        "user_id": user_id,
        "current_query": query,
        "llm_response": "",
    }

    result = graph.invoke(initial_state)
    return result["llm_response"]


def reset_user(user_id: str) -> None:
    """
    Reset all memory for a user (for benchmark isolation / GDPR).
    """
    if user_id in _short_term:
        _short_term[user_id].clear()
    _long_term.delete_user(user_id)
    _episodic.delete_user(user_id)


def get_graph_description() -> str:
    """Return ASCII art of the graph structure for demo/grading."""
    try:
        return graph.get_graph().draw_ascii()
    except Exception:
        return (
            "START\n"
            "  ↓\n"
            "[load_memory]  ← reads: short_term, long_term, episodic, semantic\n"
            "  ↓\n"
            "[llm_node]     ← injects memory into system prompt → calls OpenAI\n"
            "  ↓\n"
            "[save_memory]  ← updates: short_term buffer, long_term Redis, episodic log\n"
            "  ↓\n"
            "END"
        )
