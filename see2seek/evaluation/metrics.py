"""
metrics.py — Navigation evaluation metrics: Success Rate (SR) and SPL.

Definitions:
    SR  = fraction of episodes where the agent successfully reached the goal
          (called Stop within success_distance of goal).

    SPL = Success weighted by inverse Path Length (Anderson et al., 2018):
          SPL = (1/N) Σ S_i * L_i / max(p_i, L_i)
          where:
            S_i = 1 if episode i succeeded, else 0
            L_i = shortest path length for episode i (metres)
            p_i = actual path length taken by agent i (metres)

    SPL penalises success achieved via unnecessarily long paths. A perfect
    agent achieves SPL = SR = 1.0.

Usage:
    metrics = NavigationMetrics()
    metrics.update(success=True, path_length=3.2, shortest_path=2.5)
    metrics.update(success=False, path_length=10.0, shortest_path=2.0)
    print(metrics.summary())
    # {"sr": 0.5, "spl": 0.39, "num_episodes": 2}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class NavigationMetrics:
    """
    Accumulates navigation metrics across episodes.

    Reset between evaluation runs with .reset().
    """

    _successes:       List[int]   = field(default_factory=list)
    _path_lengths:    List[float] = field(default_factory=list)
    _shortest_paths:  List[float] = field(default_factory=list)

    def update(
        self,
        success: bool,
        path_length: float,
        shortest_path: float,
    ) -> None:
        """
        Record the outcome of one completed episode.

        Args:
            success:       True if the agent succeeded (called Stop at goal).
            path_length:   Total distance the agent actually walked (metres).
            shortest_path: Oracle shortest-path distance for this episode.
        """
        self._successes.append(int(success))
        self._path_lengths.append(path_length)
        self._shortest_paths.append(shortest_path)

    def reset(self) -> None:
        """Clear all accumulated episode data."""
        self._successes.clear()
        self._path_lengths.clear()
        self._shortest_paths.clear()

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    @property
    def sr(self) -> float:
        """Success Rate ∈ [0, 1]."""
        if not self._successes:
            return 0.0
        return sum(self._successes) / len(self._successes)

    @property
    def spl(self) -> float:
        """
        Success weighted by Path Length ∈ [0, 1].

        SPL = (1/N) Σ S_i * L_i / max(p_i, L_i)
        """
        if not self._successes:
            return 0.0
        spl_vals = []
        for s, p, l in zip(self._successes, self._path_lengths, self._shortest_paths):
            if p <= 0 and l <= 0:
                spl_vals.append(float(s))
            else:
                spl_vals.append(s * l / max(p, l))
        return sum(spl_vals) / len(spl_vals)

    @property
    def num_episodes(self) -> int:
        return len(self._successes)

    def summary(self) -> Dict[str, float]:
        """Return a dict of all metrics."""
        return {
            "sr":           round(self.sr,  4),
            "spl":          round(self.spl, 4),
            "num_episodes": float(self.num_episodes),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return f"NavigationMetrics(SR={s['sr']:.3f}, SPL={s['spl']:.3f}, N={int(s['num_episodes'])})"