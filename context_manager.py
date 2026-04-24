"""
Context Manager — Token Budget & Priority Eviction
====================================================
Manages the token budget across all 4 memory types.

Token budget from slide (% of context window):
  Short-term : 10%
  Long-term  :  4%
  Episodic   :  3%
  Semantic   :  3%
  Total memory overhead: 20% max. Above 20% → accuracy drops.

Priority eviction (when near limit):
  Trim from lowest priority first:
  4. Semantic hits   (trim first)
  3. Episodic memory
  2. Long-term facts (keep most recent N)
  1. Short-term      (trim last — most critical for coherence)

Uses tiktoken for accurate token counting (bonus point).
"""

from __future__ import annotations
from typing import Dict, Any, List
from token_counter import count_tokens as _tok, count_tokens_obj as _tok_obj, is_using_tiktoken


class ContextManager:
    """
    Token budget manager with 4-level priority eviction.

    Usage:
        ctx = ContextManager(total_context=4096)
        remaining = ctx.compute_budget(profile, episodes, hits, messages)
        state = ctx.trim(state)  # modifies state in-place copy
    """

    # Budget fractions from slide
    FRAC_SHORT_TERM: float = 0.10
    FRAC_LONG_TERM: float = 0.04
    FRAC_EPISODIC: float = 0.03
    FRAC_SEMANTIC: float = 0.03

    def __init__(self, total_context: int = 4096) -> None:
        self.total_context = total_context
        self.budget_short_term = int(total_context * self.FRAC_SHORT_TERM)
        self.budget_long_term  = int(total_context * self.FRAC_LONG_TERM)
        self.budget_episodic   = int(total_context * self.FRAC_EPISODIC)
        self.budget_semantic   = int(total_context * self.FRAC_SEMANTIC)
        self.max_memory_tokens = int(total_context * 0.20)
        self._using_tiktoken   = is_using_tiktoken()

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Accurate token count using tiktoken (with fallback)."""
        return _tok(text)

    def count_tokens_obj(self, obj: Any) -> int:
        """Count tokens in any serializable object."""
        return _tok_obj(obj)

    def total_memory_tokens(
        self,
        profile: Dict,
        episodes: List,
        semantic_hits: List,
        messages: List,
    ) -> int:
        """Sum of all memory tokens currently in state."""
        return (
            self.count_tokens_obj(profile)
            + self.count_tokens_obj(episodes)
            + self.count_tokens_obj(semantic_hits)
            + self.count_tokens_obj(messages)
        )

    def compute_budget(
        self,
        profile: Dict,
        episodes: List,
        semantic_hits: List,
        messages: List,
    ) -> int:
        """Return remaining token budget after memory usage."""
        used = self.total_memory_tokens(profile, episodes, semantic_hits, messages)
        return max(0, self.max_memory_tokens - used)

    # ------------------------------------------------------------------
    # Trim — 4-level priority eviction
    # ------------------------------------------------------------------

    def trim(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trim context when near token limit.
        Returns a new state dict (does not mutate input).

        Eviction order (lowest priority first):
          Level 4: semantic_hits (cheapest to re-retrieve)
          Level 3: episodes
          Level 2: long-term facts (keep most recent N)
          Level 1: messages (keep last 5 — most critical)
        """
        state = dict(state)  # shallow copy to avoid mutation

        remaining = self.compute_budget(
            state.get("user_profile", {}),
            state.get("episodes", []),
            state.get("semantic_hits", []),
            state.get("messages", []),
        )

        if remaining > 200:
            state["memory_budget"] = remaining
            return state  # Within budget, no trim needed

        # --- Level 4: Trim semantic hits (lowest priority) ---
        semantic_hits = list(state.get("semantic_hits", []))
        while semantic_hits and remaining < 200:
            semantic_hits.pop()  # remove last (least relevant) hit
            remaining = self.compute_budget(
                state.get("user_profile", {}),
                state.get("episodes", []),
                semantic_hits,
                state.get("messages", []),
            )
        state["semantic_hits"] = semantic_hits

        if remaining >= 200:
            state["memory_budget"] = remaining
            return state

        # --- Level 3: Trim episodic hits ---
        episodes = list(state.get("episodes", []))
        while episodes and remaining < 200:
            episodes.pop()  # remove last (least relevant) episode
            remaining = self.compute_budget(
                state.get("user_profile", {}),
                episodes,
                state.get("semantic_hits", []),
                state.get("messages", []),
            )
        state["episodes"] = episodes

        if remaining >= 200:
            state["memory_budget"] = remaining
            return state

        # --- Level 2: Trim long-term facts (keep 3 most recent) ---
        profile = dict(state.get("user_profile", {}))
        facts = dict(profile.get("facts", {}))
        if len(facts) > 3:
            # Keep only the last 3 facts (dict preserves insertion order in Python 3.7+)
            recent_keys = list(facts.keys())[-3:]
            profile["facts"] = {k: facts[k] for k in recent_keys}
            state["user_profile"] = profile
            remaining = self.compute_budget(
                profile,
                state.get("episodes", []),
                state.get("semantic_hits", []),
                state.get("messages", []),
            )

        if remaining >= 200:
            state["memory_budget"] = remaining
            return state

        # --- Level 1: Trim short-term messages (keep last 5) ---
        messages = list(state.get("messages", []))
        if len(messages) > 5:
            state["messages"] = messages[-5:]

        state["memory_budget"] = max(
            0,
            self.compute_budget(
                state.get("user_profile", {}),
                state.get("episodes", []),
                state.get("semantic_hits", []),
                state.get("messages", []),
            ),
        )
        return state

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(
        self,
        profile: Dict,
        episodes: List,
        semantic_hits: List,
        messages: List,
    ) -> Dict[str, int]:
        """Return token breakdown per memory type for benchmark report."""
        return {
            "short_term_tokens": self.count_tokens_obj(messages),
            "long_term_tokens": self.count_tokens_obj(profile),
            "episodic_tokens": self.count_tokens_obj(episodes),
            "semantic_tokens": self.count_tokens_obj(semantic_hits),
            "total_memory_tokens": self.total_memory_tokens(
                profile, episodes, semantic_hits, messages
            ),
            "budget_remaining": self.compute_budget(
                profile, episodes, semantic_hits, messages
            ),
            "max_memory_budget": self.max_memory_tokens,
        }

    def __repr__(self) -> str:
        return (
            f"ContextManager(total={self.total_context}, "
            f"memory_cap={self.max_memory_tokens} tokens)"
        )
