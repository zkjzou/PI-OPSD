"""Backward-compat shim — the PI construction package moved to ``opsd.pi`` (Phase 3 refactor).

Existing importers (``from opsd.trainer.pi import build_constructor`` etc.) keep working
via these re-exports; the constructor/artifact/analyzer/online modules now live under
``opsd.pi.*``. ``score_trajectories`` and ``turn_channel_kl`` remain in this package.
"""

from opsd.pi import (  # noqa: F401
    HintConstructor,
    HintRecord,
    StaticHintConstructor,
    TrajView,
    build_constructor,
    load_hint_artifact,
    load_task_hints,
    save_hint_artifact,
)

__all__ = [
    "HintConstructor",
    "HintRecord",
    "TrajView",
    "StaticHintConstructor",
    "load_hint_artifact",
    "load_task_hints",
    "save_hint_artifact",
    "build_constructor",
]
