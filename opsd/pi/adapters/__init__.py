"""Benchmark schema adapters for PI scoring/analysis.

``get_adapter(name)`` returns the registered ``SchemaAdapter`` for a benchmark (ALFWorld and AppWorld
today, plus τ²-bench; add a module + ``register_adapter`` for a new benchmark). ``skyrl_eval_dump``
is a loader/reconstructor for SkyRL's own eval-dump format (``--input-format skyrl_eval`` in the scorers)
and is imported on demand (not registered as a schema adapter).
"""

from opsd.pi.adapters.alfworld import AlfworldAdapter
from opsd.pi.adapters.appworld import AppworldAdapter
from opsd.pi.adapters.base import SchemaAdapter
from opsd.pi.adapters.tau2 import Tau2Adapter
from opsd.pi.adapters.tokenization import (
    ModelTokenSpec,
    get_token_spec,
    resolve_im_end_id,
)

_ADAPTERS = {}


def register_adapter(adapter) -> None:
    """Register a SchemaAdapter instance under ``adapter.name``."""
    _ADAPTERS[adapter.name] = adapter


def get_adapter(name: str = "alfworld") -> SchemaAdapter:
    """Return the registered SchemaAdapter for a benchmark. KeyError if unknown."""
    return _ADAPTERS[name]


register_adapter(AlfworldAdapter())
register_adapter(AppworldAdapter())
register_adapter(Tau2Adapter())

__all__ = [
    "SchemaAdapter",
    "AlfworldAdapter",
    "AppworldAdapter",
    "Tau2Adapter",
    "ModelTokenSpec",
    "get_token_spec",
    "resolve_im_end_id",
    "register_adapter",
    "get_adapter",
]
