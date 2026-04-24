"""
Memory Router — Query Intent Classifier
=========================================
Determines which memory type(s) to load for a given user query.
Based on slide: "chọn memory type phù hợp dựa trên query intent —
user preference vs factual recall vs experience recall"

Classification logic uses keyword matching (fast, deterministic).
Returns a list of memory types that should be loaded and injected.
"""

from __future__ import annotations
from typing import List, Dict


# ---------------------------------------------------------------------------
# Keyword sets for each memory type
# ---------------------------------------------------------------------------

_PREF_KEYWORDS: frozenset = frozenset([
    # Vietnamese
    "thích", "không thích", "prefer", "muốn", "yêu thích", "ghét",
    "ngôn ngữ", "style", "cách viết", "format",
    # English
    "like", "dislike", "favorite", "prefer",
])

_EXPERIENCE_KEYWORDS: frozenset = frozenset([
    # Vietnamese
    "lần trước", "trước đây", "đã làm", "đã fix", "nhớ không",
    "bạn có nhớ", "lần này giống", "trước đó", "debug", "fix", "lỗi",
    "giải quyết", "đã gặp", "kinh nghiệm",
    # English
    "last time", "before", "previous", "remember", "past", "history",
    "fixed", "solved", "error",
])

_SEMANTIC_KEYWORDS: frozenset = frozenset([
    # Vietnamese
    "là gì", "tại sao", "cách", "giải thích", "định nghĩa", "hoạt động",
    "khác nhau", "so sánh", "khi nào", "làm sao", "ví dụ",
    # English
    "what is", "how", "explain", "why", "when", "difference", "example",
    # Technical topics that always benefit from semantic retrieval
    "python", "numpy", "pandas", "sklearn", "async", "await", "asyncio",
    "machine learning", "deep learning", "neural", "overfitting",
    "regularization", "dropout", "attention", "transformer", "bert",
    "pytorch", "tensorflow", "cuda", "huggingface",
    "list comprehension", "dictionary", "decorator", "metaclass",
    "generator", "iterator", "lambda",
])

_PROFILE_RECALL_KEYWORDS: frozenset = frozenset([
    # Vietnamese
    "tên tôi", "tôi là", "tôi đang", "level của tôi",
    "bạn biết gì về tôi", "thông tin của tôi", "profile của tôi",
    "tôi dị ứng", "sở thích của tôi",
    # English
    "my name", "who am i", "my level", "what do you know about me",
])


class MemoryRouter:
    """
    Route a user query to the appropriate memory type(s).

    Returns a list from: ["short_term", "long_term", "episodic", "semantic"]
    short_term is always included.
    """

    def classify(self, query: str) -> List[str]:
        """
        Classify query and return relevant memory types.

        Strategy:
        - short_term: always (recent conversation context)
        - long_term: preference/profile queries OR profile recall
        - episodic: experience/past task queries
        - semantic: factual/knowledge queries
        - If nothing specific matches → load all (comprehensive fallback)
        """
        q = query.lower()
        memory_types: List[str] = ["short_term"]

        words = set(q.split())

        # Check preference queries
        if _PREF_KEYWORDS & words or any(kw in q for kw in _PREF_KEYWORDS):
            if "long_term" not in memory_types:
                memory_types.append("long_term")

        # Check profile recall (user asking agent to recall stored profile)
        if any(kw in q for kw in _PROFILE_RECALL_KEYWORDS):
            if "long_term" not in memory_types:
                memory_types.append("long_term")

        # Check experience/episodic queries
        if any(kw in q for kw in _EXPERIENCE_KEYWORDS):
            if "episodic" not in memory_types:
                memory_types.append("episodic")

        # Check semantic/knowledge queries
        if any(kw in q for kw in _SEMANTIC_KEYWORDS):
            if "semantic" not in memory_types:
                memory_types.append("semantic")

        # Fallback: load all if no specific match
        # (better to over-retrieve than miss relevant context)
        if len(memory_types) == 1:
            memory_types.extend(["long_term", "episodic", "semantic"])

        return memory_types

    def get_primary_intent(self, query: str) -> str:
        """
        Return the single most likely intent label for logging/debugging.

        Labels:
          preference_query   → user expressing or asking about preferences
          experience_recall  → user asking about past tasks/episodes
          factual_recall     → user asking factual/knowledge question
          general            → multiple intents or unclassified
        """
        types = self.classify(query)
        type_set = set(types)

        if "episodic" in type_set and "semantic" not in type_set:
            return "experience_recall"
        if "semantic" in type_set and "episodic" not in type_set:
            return "factual_recall"
        if "long_term" in type_set and len(type_set) == 2:
            return "preference_query"
        return "general"

    def describe(self, query: str) -> Dict[str, object]:
        """Return full classification breakdown for debugging."""
        types = self.classify(query)
        return {
            "query": query[:80],
            "memory_types": types,
            "primary_intent": self.get_primary_intent(query),
        }
