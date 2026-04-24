"""
Short-term Memory — Context Window Buffer
==========================================
Equivalent to "Working Memory" / RAM in the brain analogy (slide).
- Fast, temporary, limited capacity (~128K tokens for production).
- Implements sliding window strategy: system + summary + last K turns.
- Uses tiktoken for accurate token counting (bonus point).
- Auto-trim when approaching token budget (10% of context from slide).
"""

from __future__ import annotations
from typing import List, Dict, Any
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from token_counter import count_tokens as _count_tok


class ShortTermMemory:
    """
    Sliding window conversation buffer.

    3 trim strategies (from slide):
      - Buffer   : keep all messages (simple, hits limit ~50 turns)
      - Summary  : LLM summarize old history (stable but extra LLM call)
      - Sliding  : system + summary + last K turns (BEST, used here)
    """

    def __init__(self, max_messages: int = 20, max_tokens: int = 2000) -> None:
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.messages: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Core write operations
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        """Append a message and trim if over budget."""
        token_count = self._count_tokens(content)
        self.messages.append({
            "role": role,
            "content": content,
            "tokens": token_count,
        })
        self._trim_if_needed()

    # ------------------------------------------------------------------
    # Core read operations
    # ------------------------------------------------------------------

    def get_recent(self, k: int = 10) -> List[Dict[str, Any]]:
        """Return last k messages (sliding window)."""
        return self.messages[-k:] if len(self.messages) >= k else list(self.messages)

    def get_all(self) -> List[Dict[str, Any]]:
        """Return all buffered messages."""
        return list(self.messages)

    # ------------------------------------------------------------------
    # Token accounting
    # ------------------------------------------------------------------

    def count_tokens(self) -> int:
        """Total tokens currently in buffer."""
        return sum(m.get("tokens", 0) for m in self.messages)

    def _count_tokens(self, text: str) -> int:
        return _count_tok(text)

    # ------------------------------------------------------------------
    # Trim — sliding window strategy
    # ------------------------------------------------------------------

    def _trim_if_needed(self) -> None:
        """
        Evict oldest messages first when over budget.
        Keeps most-recent turns to preserve conversational coherence.
        """
        # Trim by token count
        while self.count_tokens() > self.max_tokens and len(self.messages) > 2:
            self.messages.pop(0)

        # Trim by message count
        while len(self.messages) > self.max_messages:
            self.messages.pop(0)

    def trim_to_budget(self, max_tokens: int) -> None:
        """Explicitly trim buffer to a given token budget (called by ContextManager)."""
        while self.count_tokens() > max_tokens and len(self.messages) > 2:
            self.messages.pop(0)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all messages (e.g., new session)."""
        self.messages = []

    def format_for_prompt(self) -> str:
        """Format recent messages as readable text for prompt injection."""
        if not self.messages:
            return "No recent conversation."
        lines = []
        for m in self.get_recent(k=10):
            role_label = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"{role_label}: {m['content']}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.messages)

    def __repr__(self) -> str:
        return (
            f"ShortTermMemory(messages={len(self.messages)}, "
            f"tokens={self.count_tokens()}/{self.max_tokens})"
        )
