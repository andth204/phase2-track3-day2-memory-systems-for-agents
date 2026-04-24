"""
Long-term Memory — Persistent Cross-Session Store (Redis)
==========================================================
Equivalent to "Declarative Memory" / Hard disk in the brain analogy (slide).
- Uses fakeredis (same API as redis-py, no server needed for lab).
- Stores user preferences + facts as key-value pairs with TTL.
- TTL policy from slide: prefs=90d, facts=30d, sessions=7d.
- Conflict resolution: RECENCY WINS — new fact always overwrites old.
- GDPR: delete_user() removes ALL entries for a user (Right to be Forgotten).
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import fakeredis


class LongTermMemory:
    """
    Redis-backed persistent profile storage.

    Storage layout (Redis keys):
      prefs:{user_id}:{key}     → user preferences (language, style, etc.)
      facts:{user_id}:{key}     → factual data (name, level, allergy, etc.)
      sessions:{user_id}        → list of recent session summaries
    """

    # TTL from slide
    TTL_PREFS: int = 90 * 24 * 3600    # 90 days
    TTL_FACTS: int = 30 * 24 * 3600    # 30 days
    TTL_SESSIONS: int = 7 * 24 * 3600  # 7 days

    def __init__(self) -> None:
        # fakeredis: production-compatible interface, no server required
        self.redis = fakeredis.FakeRedis(decode_responses=True)

    # ------------------------------------------------------------------
    # Preferences (language, response style, topics liked/disliked)
    # ------------------------------------------------------------------

    def set_preference(self, user_id: str, key: str, value: str) -> None:
        """Store a user preference. Overwrites existing (recency wins)."""
        self.redis.set(f"prefs:{user_id}:{key}", value, ex=self.TTL_PREFS)

    def get_preference(self, user_id: str, key: str) -> Optional[str]:
        return self.redis.get(f"prefs:{user_id}:{key}")

    def get_all_preferences(self, user_id: str) -> Dict[str, str]:
        """Return all preferences for a user."""
        pattern = f"prefs:{user_id}:*"
        keys = self.redis.keys(pattern)
        result: Dict[str, str] = {}
        for key in keys:
            pref_key = key.split(":")[-1]
            value = self.redis.get(key)
            if value is not None:
                result[pref_key] = value
        return result

    # ------------------------------------------------------------------
    # Facts (name, skill level, allergy, errors encountered, etc.)
    # CONFLICT HANDLING: recency wins — SET always overwrites same key
    # ------------------------------------------------------------------

    def set_fact(self, user_id: str, key: str, value: str) -> None:
        """
        Store a factual datum about the user.
        RECENCY WINS: if same key already exists, new value overwrites old.
        This implements the conflict resolution policy from slide.
        """
        self.redis.set(f"facts:{user_id}:{key}", value, ex=self.TTL_FACTS)

    def get_fact(self, user_id: str, key: str) -> Optional[str]:
        return self.redis.get(f"facts:{user_id}:{key}")

    def get_all_facts(self, user_id: str) -> Dict[str, str]:
        """Return all facts for a user."""
        pattern = f"facts:{user_id}:*"
        keys = self.redis.keys(pattern)
        result: Dict[str, str] = {}
        for key in keys:
            fact_key = key.split(":")[-1]
            value = self.redis.get(key)
            if value is not None:
                result[fact_key] = value
        return result

    # ------------------------------------------------------------------
    # Composite profile
    # ------------------------------------------------------------------

    def get_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Return full profile: preferences + facts.
        Called by load_memory node in LangGraph agent.
        """
        return {
            "preferences": self.get_all_preferences(user_id),
            "facts": self.get_all_facts(user_id),
        }

    def update_profile_from_dict(self, user_id: str, data: Dict[str, Any]) -> None:
        """
        Bulk update profile from a parsed extraction dict.
        Recency wins on every field — simply SET overwrites.

        Expected format:
            {
              "preferences": {"language": "Python", "style": "concise"},
              "facts": {"name": "Minh", "level": "beginner", "allergy": "đậu nành"}
            }
        """
        for key, value in data.get("preferences", {}).items():
            if key and value:
                self.set_preference(user_id, str(key).strip(), str(value).strip())

        for key, value in data.get("facts", {}).items():
            if key and value:
                self.set_fact(user_id, str(key).strip(), str(value).strip())

    # ------------------------------------------------------------------
    # Session history (recent session summaries, List in Redis)
    # ------------------------------------------------------------------

    def add_session_summary(self, user_id: str, summary: str) -> None:
        """Push session summary to list. Keep last 10."""
        list_key = f"sessions:{user_id}"
        self.redis.lpush(list_key, summary)
        self.redis.ltrim(list_key, 0, 9)  # keep last 10
        self.redis.expire(list_key, self.TTL_SESSIONS)

    def get_session_history(self, user_id: str) -> List[str]:
        return self.redis.lrange(f"sessions:{user_id}", 0, -1)

    # ------------------------------------------------------------------
    # GDPR: Right to be Forgotten
    # ------------------------------------------------------------------

    def delete_user(self, user_id: str) -> int:
        """
        Delete ALL data for a user (GDPR Right to be Forgotten).
        Returns number of keys deleted.
        """
        deleted = 0
        for pattern in [
            f"prefs:{user_id}:*",
            f"facts:{user_id}:*",
            f"sessions:{user_id}",
        ]:
            keys = self.redis.keys(pattern)
            for key in keys:
                self.redis.delete(key)
                deleted += 1
        return deleted

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def format_for_prompt(self, user_id: str) -> str:
        """Format profile as text for system prompt injection."""
        profile = self.get_profile(user_id)
        prefs = profile.get("preferences", {})
        facts = profile.get("facts", {})
        if not prefs and not facts:
            return "No user profile available."
        lines = []
        if prefs:
            lines.append(f"Preferences: {', '.join(f'{k}={v}' for k, v in prefs.items())}")
        if facts:
            lines.append(f"Known facts: {', '.join(f'{k}={v}' for k, v in facts.items())}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"LongTermMemory(backend=fakeredis)"
