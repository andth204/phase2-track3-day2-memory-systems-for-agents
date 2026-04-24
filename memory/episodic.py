"""
Episodic Memory — Learning from Past Trajectories
===================================================
Stores (task, trajectory, outcome, reflection) tuples — Voyager-style.
- Backed by JSON file (persistent across runs).
- Retrieval: keyword-overlap similarity search with importance decay.
- Management: LRU eviction, importance decay, consolidation (from slide).
- Agent learns: "approach X failed for reason Y in similar task".

Reference (slide): Episodic memory = personal, temporary→persistent
"""

from __future__ import annotations
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class EpisodicMemory:
    """
    JSON-file based episodic memory store.

    Episode schema:
        id          : int — sequential identifier
        user_id     : str — owner of this episode
        task        : str — what task was being attempted
        trajectory  : str — steps taken / approaches tried
        outcome     : str — what happened, what worked
        reflection  : str — key insight or lesson learned
        timestamp   : str — ISO 8601
        importance_score : float — decays over time, boosts on retrieval
    """

    def __init__(self, log_file: str = "data/episodes.json") -> None:
        self.log_file = log_file
        self.episodes: List[Dict[str, Any]] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load episodes from disk. Silently skip if file missing."""
        if os.path.exists(self.log_file):
            with open(self.log_file, "r", encoding="utf-8") as f:
                self.episodes = json.load(f)

    def _save(self) -> None:
        """Persist episodes to disk after every write."""
        os.makedirs(os.path.dirname(self.log_file) or ".", exist_ok=True)
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(self.episodes, f, ensure_ascii=False, indent=2)

    def load_seeds(self, seed_file: str) -> None:
        """Load seed episodes from a separate file without overwriting log."""
        if not os.path.exists(seed_file):
            return
        with open(seed_file, "r", encoding="utf-8") as f:
            seeds = json.load(f)
        # Only add seeds if they don't already exist (idempotent)
        existing_ids = {ep["id"] for ep in self.episodes}
        for seed in seeds:
            if seed["id"] not in existing_ids:
                self.episodes.append(seed)
        self._save()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def log_episode(
        self,
        task: str,
        trajectory: str,
        outcome: str,
        reflection: str,
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """
        Log a new episode tuple after a task is completed.
        Only persist after task completion to avoid inconsistent state (slide).
        """
        episode: Dict[str, Any] = {
            "id": self._next_id(),
            "user_id": user_id,
            "task": task,
            "trajectory": trajectory,
            "outcome": outcome,
            "reflection": reflection,
            "timestamp": datetime.now().isoformat(),
            "importance_score": 1.0,
        }
        self.episodes.append(episode)
        self._save()
        return episode

    def _next_id(self) -> int:
        return max((ep["id"] for ep in self.episodes), default=0) + 1

    # ------------------------------------------------------------------
    # Retrieval — keyword overlap + importance score
    # ------------------------------------------------------------------

    def search_similar(
        self, query: str, k: int = 3, user_id: str = "default"
    ) -> List[Dict[str, Any]]:
        """
        Return top-k episodes relevant to query.

        Scoring: keyword overlap across all episode fields, weighted by
        importance_score (implements importance decay from slide).
        """
        query_words = set(query.lower().split())
        scored: List[tuple] = []

        for ep in self.episodes:
            # Filter by user (or "default" = shared pool)
            if user_id != "default" and ep.get("user_id") not in (user_id, "demo_user"):
                continue

            # Build text from all fields
            ep_text = " ".join([
                ep.get("task", ""),
                ep.get("trajectory", ""),
                ep.get("outcome", ""),
                ep.get("reflection", ""),
            ]).lower()
            ep_words = set(ep_text.split())

            overlap = len(query_words & ep_words)
            if overlap == 0:
                continue

            # Weight by importance decay score
            score = overlap * ep.get("importance_score", 1.0)
            scored.append((score, ep))

        # Sort descending, return top-k
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [ep for _, ep in scored[:k]]

        # Boost importance score on retrieval (recently used = more important)
        for ep in top:
            ep["importance_score"] = min(ep.get("importance_score", 1.0) + 0.1, 2.0)
        if top:
            self._save()

        return top

    def get_all(self, user_id: str = "default") -> List[Dict[str, Any]]:
        """Return all episodes for a user."""
        if user_id == "default":
            return list(self.episodes)
        return [ep for ep in self.episodes if ep.get("user_id") == user_id]

    # ------------------------------------------------------------------
    # Management: LRU eviction, importance decay, consolidation
    # ------------------------------------------------------------------

    def apply_importance_decay(self, decay_factor: float = 0.95) -> None:
        """
        Decay importance scores over time (slide: importance decay).
        Call periodically (e.g., at session start).
        """
        for ep in self.episodes:
            ep["importance_score"] = max(
                ep.get("importance_score", 1.0) * decay_factor, 0.1
            )
        self._save()

    def evict_lru(self, max_episodes: int = 100) -> int:
        """
        LRU eviction: remove lowest-importance episodes when over capacity.
        Returns number of episodes removed.
        """
        if len(self.episodes) <= max_episodes:
            return 0
        # Sort by importance score ascending (least important first)
        self.episodes.sort(key=lambda x: x.get("importance_score", 1.0))
        removed = len(self.episodes) - max_episodes
        self.episodes = self.episodes[removed:]
        self._save()
        return removed

    # ------------------------------------------------------------------
    # GDPR
    # ------------------------------------------------------------------

    def delete_user(self, user_id: str) -> int:
        """Remove all episodes for a user (Right to be Forgotten)."""
        before = len(self.episodes)
        self.episodes = [
            ep for ep in self.episodes if ep.get("user_id") != user_id
        ]
        self._save()
        return before - len(self.episodes)

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def format_for_prompt(self, episodes: List[Dict[str, Any]]) -> str:
        """Format episodes as text for system prompt injection."""
        if not episodes:
            return "No relevant past episodes."
        lines = []
        for ep in episodes:
            lines.append(
                f"- Task: {ep.get('task', '')}\n"
                f"  Outcome: {ep.get('outcome', '')}\n"
                f"  Lesson: {ep.get('reflection', '')}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.episodes)

    def __repr__(self) -> str:
        return f"EpisodicMemory(episodes={len(self.episodes)}, file={self.log_file})"
