"""PI scoring signals + selector-quality metrics (model/benchmark-agnostic).

``metrics`` is the single source of truth for the ranking/curation metrics that were previously
copy-pasted across ``opsd/trainer/pi/score_trajectories.py`` and several offline PI-ranking analysis scripts.
"""

from opsd.pi.signals import metrics  # noqa: F401

__all__ = ["metrics"]
